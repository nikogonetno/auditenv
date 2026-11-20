"""reset/step contract of case §04 (role B).

    reset(seed, config) -> (observation, info)
    step(action)        -> (observation, reward, terminated, truncated, info)

An action is a JSON object {"action": <name>, ...arguments}:
    list_records   table, cursor (optional), limit (optional, 1..50, default 50)
    get_record     table, id
    get_document   doc_id
    calculator     expression
    submit         findings  (the findings.json object of case §05)

Every step counts, including errors and submit; at most 100. An observation holds only
the result or a short error code and the steps left. Reward is 0 until the episode ends.
In "train" mode the terminal reward goes back to the policy; in "eval" mode the policy
always sees 0 and empty info, and the grade stays in grader_log.
"""
import copy

from .calculator import CalculatorError, evaluate
from .episode import TABLES, load_episode
from .grader import MAX_STEPS, parse_truth, score
from .grader.matcher import normalize_id
from .submission import InvalidSubmission, check_submission

PAGE_LIMIT = 50
MAX_ID_LENGTH = 200
MODES = ("train", "eval")
CONFIG_KEYS = {"episode_dir"}
TASK = (
    "You audit one company for one reporting period. Read records with list_records, get_record and "
    "get_document, check arithmetic with calculator, then call submit once with findings.json: "
    "{episode_id, findings: [{type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}]}. "
    "Types: duplicate_invoice, payment_allocation, fx_rate, period_cutoff. At most 30 findings, 10 evidence ids each. "
    "Every action uses one of 100 steps; submit ends the episode."
)
ACTIONS = {
    "list_records": "list_records(table, cursor=null, limit=50)",
    "get_record": "get_record(table, id)",
    "get_document": "get_document(doc_id)",
    "calculator": "calculator(expression)",
    "submit": "submit(findings)",
}


class ActionError(Exception):
    """A bounded error code for the observation. Consumes the step like any action."""


class AuditEnv:
    def __init__(self, mode="train"):
        if mode not in MODES:
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self.grader_log = []
        self._episode = None

    # ------------------------------------------------------------------ contract
    def reset(self, seed=None, config=None):
        """Start a fresh episode. config {"episode_dir": path} loads a prepared episode."""
        config = {} if config is None else config
        if type(config) is not dict:
            raise ValueError("config must be an object")
        unknown = sorted(set(config) - CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unknown config key: {unknown[0]}")
        if "episode_dir" not in config:
            raise ValueError("config.episode_dir is required until the generator lands")
        episode = load_episode(config["episode_dir"])  # EpisodeError is a ValueError
        # Truth is read and checked once, here: a file changed later (even by a policy that
        # can reach the disk) cannot change the grade. ValueError on a malformed truth.
        truth = episode.load_truth()
        parse_truth(truth, episode.aliases)
        self._episode = episode
        self._truth = truth
        self._steps = 0
        self._done = False
        self._terminated = self._truncated = False
        observation = {
            "task": TASK,
            "manifest": copy.deepcopy(episode.manifest),
            "policy": copy.deepcopy(episode.policy),
            "tables": {name: len(episode.tables[name]) for name in TABLES},
            "actions": dict(ACTIONS),
            "steps_left": MAX_STEPS,
        }
        return observation, {}

    def step(self, action):
        if self._episode is None:
            return {"error": "reset_required", "steps_left": 0}, 0.0, False, False, {}
        if self._done:
            # E-05: nothing changes after the end, and no step is spent.
            return {"error": "episode_over", "steps_left": MAX_STEPS - self._steps}, 0.0, \
                self._terminated, self._truncated, {}
        self._steps += 1
        if type(action) is dict and action.get("action") == "submit":
            return self._submit(action)
        try:
            if type(action) is not dict:
                raise ActionError("bad_action")  # also every policy line that does not decode
            body = {"result": self._dispatch(action.get("action"), action)}
        except ActionError as exc:
            body = {"error": str(exc)}
        if self._steps >= MAX_STEPS:
            return self._finish_empty(body)
        body["steps_left"] = MAX_STEPS - self._steps
        return body, 0.0, False, False, {}

    def truncate(self):
        """Wall-clock limit (60 s per episode): end with an empty answer charged for 100 steps."""
        if self._episode is None or self._done:
            return self.step({"action": "noop"})
        self._steps = MAX_STEPS
        return self._finish_empty({})

    # ------------------------------------------------------------------ actions
    def _dispatch(self, name, action):
        if name == "list_records":
            return self._list_records(action)
        if name == "get_record":
            return self._get_record(action)
        if name == "get_document":
            return self._get_document(action)
        if name == "calculator":
            expression = action.get("expression")
            try:
                return {"value": evaluate(expression)}
            except CalculatorError as exc:
                raise ActionError(str(exc)) from None
        raise ActionError("unknown_action")

    def _table(self, action):
        table = action.get("table")
        if type(table) is not str or table not in TABLES:
            raise ActionError("unknown_table")
        return table

    def _list_records(self, action):
        table = self._table(action)
        limit = action.get("limit", PAGE_LIMIT)
        if limit is None:
            limit = PAGE_LIMIT
        if type(limit) is not int or not 1 <= limit <= PAGE_LIMIT:
            raise ActionError("bad_limit")
        rows = self._episode.tables[table]
        offset = self._parse_cursor(action.get("cursor"), table, len(rows))
        page = rows[offset:offset + limit]
        end = offset + len(page)
        return {
            "table": table,
            "records": copy.deepcopy(page),
            "next_cursor": self._cursor(table, end) if end < len(rows) else None,
        }

    def _cursor(self, table, offset):
        return f"c1.{self._episode.episode_id}.{table}.{offset}"

    def _parse_cursor(self, cursor, table, total):
        if cursor is None:
            return 0
        if type(cursor) is not str or len(cursor) > MAX_ID_LENGTH:
            raise ActionError("bad_cursor")
        parts = cursor.split(".")
        if (len(parts) != 4 or parts[0] != "c1" or parts[1] != self._episode.episode_id
                or parts[2] != table or not parts[3].isdigit() or not parts[3].isascii()):
            raise ActionError("bad_cursor")
        offset = int(parts[3])
        if offset > total or (offset == total and total):
            raise ActionError("bad_cursor")
        return offset

    def _clean_id(self, value):
        if type(value) is not str or not value or len(value) > MAX_ID_LENGTH:
            raise ActionError("bad_arguments")
        return normalize_id(value)

    def _get_record(self, action):
        table = self._table(action)
        rid = self._clean_id(action.get("id"))
        record = self._episode.index[table].get(rid)
        if record is None:
            raise ActionError("not_found")
        return {"table": table, "record": copy.deepcopy(record)}

    def _get_document(self, action):
        doc_id = self._clean_id(action.get("doc_id"))
        found = self._episode.documents.get(doc_id)
        if found is None:
            raise ActionError("not_found")
        table, record = found
        return {"table": table, "document": copy.deepcopy(record)}

    # ------------------------------------------------------------------ end of episode
    def _submit(self, action):
        inspection_steps = self._steps - 1
        episode = self._episode
        try:
            findings = check_submission(action.get("findings"), episode.episode_id)
        except InvalidSubmission as exc:
            entry = {"episode_id": episode.episode_id, "mode": self.mode, "end": "submit",
                     "inspection_steps": inspection_steps, "error": str(exc), "reward": -1.0}
            return self._end({"result": {"status": "submitted"}}, entry, terminated=True)
        result = score(findings, self._truth, inspection_steps, episode.aliases)
        entry = {"episode_id": episode.episode_id, "mode": self.mode, "end": "submit", **result}
        return self._end({"result": {"status": "submitted"}}, entry, terminated=True)

    def _finish_empty(self, body):
        episode = self._episode
        result = score([], self._truth, MAX_STEPS, episode.aliases)
        entry = {"episode_id": episode.episode_id, "mode": self.mode, "end": "truncated", **result}
        return self._end(body, entry, terminated=False)

    def _end(self, body, entry, terminated):
        self._done = True
        self._terminated, self._truncated = terminated, not terminated
        self.grader_log.append(entry)
        body["steps_left"] = MAX_STEPS - self._steps
        if self.mode == "eval":
            return body, 0.0, terminated, not terminated, {}
        info = {"grading": {k: entry[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "error") if k in entry}}
        return body, entry["reward"], terminated, not terminated, info
