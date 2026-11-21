"""Jury rehearsal: 20 contract checks and 20 reward checks, as the case §10 blocks count them.

    python3 qa/rehearsal/blocks.py [--block contract,reward]

Written from case §04, §05 and SPEC §4, §5 against the public AuditEnv only, on the rehearsal episodes
(9007: C1 + C2 + fx, 5 issues; 9017: 12 issues of all four types; 9002: clean). The case §10 score of a
block is 25 x passed / 20. Needs `python3 qa/rehearsal/part_a.py --steps build`.
Writes runs/rehearsal/blocks.json, exit 1 if any check fails.
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, emit, episode_dir, oracle_findings, truth_of, write_json  # noqa: E402

C1_SEED, FULL_SEED, CLEAN_SEED = 9007, 9017, 9002
MANIFEST_KEYS = ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                 "counts", "documents", "rounding")
TABLES = ("ledger", "invoices", "payments", "fx")


def env_for(seed, mode="train"):
    from auditenv.env import AuditEnv
    env = AuditEnv(mode)
    obs, info = env.reset(config={"episode_dir": str(episode_dir(seed))})
    return env, obs, info


def jsonable(value):
    return json.loads(json.dumps(value)) == value


def page_all(env, table):
    records, cursor, steps = [], None, 0
    while True:
        obs, *_ = env.step({"action": "list_records", "table": table, "cursor": cursor, "limit": 50})
        steps += 1
        result = obs["result"]
        records += result["records"]
        cursor = result["next_cursor"]
        if cursor is None:
            return records, steps


def read_public(seed, table):
    import csv
    base = episode_dir(seed) / "public"
    if table in ("ledger", "fx"):
        with open(base / f"{table}.csv", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    with open(base / f"{table}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------------------------- contract
def contract_checks():
    checks = []

    def check(name):
        def wrap(fn):
            checks.append((name, fn))
            return fn
        return wrap

    @check("C01 reset gives (observation, info): task, manifest, policy, tables, actions, 100 steps")
    def _():
        from auditenv.env import AuditEnv
        out = AuditEnv("eval").reset(config={"episode_dir": str(episode_dir(C1_SEED))})
        obs, info = out
        need = {"task", "manifest", "policy", "tables", "actions", "steps_left"}
        return [] if (type(out) is tuple and len(out) == 2 and need <= set(obs) and obs["steps_left"] == 100
                      and info == {} and jsonable(obs)) else [f"keys {sorted(obs)}, info {info}"]

    @check("C02 manifest carries every case §03 field")
    def _():
        _, obs, _ = env_for(C1_SEED)
        missing = [k for k in MANIFEST_KEYS if k not in obs["manifest"]]
        return [f"missing {missing}"] if missing else []

    @check("C03 step gives five JSON values: observation, float reward, bool, bool, dict")
    def _():
        env, *_ = env_for(C1_SEED)
        out = env.step({"action": "list_records", "table": "ledger", "cursor": None, "limit": 3})
        ok = (len(out) == 5 and type(out[1]) is float and type(out[2]) is bool and type(out[3]) is bool
              and type(out[4]) is dict and jsonable(list(out)))
        return [] if ok else [repr(out)[:200]]

    @check("C04 a page holds at most 50 rows, 50 by default; limit 51 is refused and spends a step")
    def _():
        env, *_ = env_for(FULL_SEED)
        obs, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": None})
        over, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": None, "limit": 51})
        p = []
        if len(obs["result"]["records"]) != 50:
            p.append(f"default page {len(obs['result']['records'])}")
        if "error" not in over or over["steps_left"] != 98:
            p.append(f"limit 51: {over}")
        return p

    @check("C05 cursors are stable and paging gives every row exactly once")
    def _():
        p = []
        for table in TABLES:
            env, obs, _ = env_for(FULL_SEED)
            first, *_ = env.step({"action": "list_records", "table": table, "cursor": None, "limit": 7})
            cursor = first["result"]["next_cursor"]
            a, *_ = env.step({"action": "list_records", "table": table, "cursor": cursor, "limit": 7})
            b, *_ = env.step({"action": "list_records", "table": table, "cursor": cursor, "limit": 7})
            if a["result"] != b["result"]:
                p.append(f"{table}: one cursor, two pages")
            env, obs, _ = env_for(FULL_SEED)
            records, _ = page_all(env, table)
            ids = [json.dumps(r, sort_keys=True) for r in records]
            if len(ids) != len(set(ids)) or len(ids) != obs["tables"][table]:
                p.append(f"{table}: {len(ids)} rows paged, {len(set(ids))} distinct, {obs['tables'][table]} declared")
        return p

    @check("C06 get_record with and without the table prefix, get_document by doc_id (case §04)")
    def _():
        env, *_ = env_for(C1_SEED)
        inv, pay = read_public(C1_SEED, "invoices")[0], read_public(C1_SEED, "payments")[0]
        a, *_ = env.step({"action": "get_record", "table": "invoices", "id": inv["invoice_id"]})
        b, *_ = env.step({"action": "get_record", "table": "invoices", "id": "invoice/" + inv["invoice_id"]})
        c, *_ = env.step({"action": "get_document", "doc_id": inv["doc_id"]})
        d, *_ = env.step({"action": "get_document", "doc_id": "doc/" + inv["doc_id"]})
        e, *_ = env.step({"action": "get_document", "doc_id": pay["doc_id"]})
        p = []
        if "result" not in a or a["result"] != b["result"]:
            p.append(f"get_record {a} vs {b}"[:200])
        if "result" not in c or c.get("result") != d.get("result") or "result" not in e:
            p.append(f"get_document {c} vs {d}, payment {e}"[:200])
        return p

    @check("C06b SPEC §3.3, §4.2: get_document also takes invoice_id and payment_id (not counted, case §04 asks doc_id)")
    def _():
        env, *_ = env_for(C1_SEED)
        inv, pay = read_public(C1_SEED, "invoices")[0], read_public(C1_SEED, "payments")[0]
        out = [env.step({"action": "get_document", "doc_id": x})[0] for x in (inv["invoice_id"], pay["payment_id"])]
        return [] if all("result" in o for o in out) else [f"get_document by invoice_id, payment_id: {out}"]

    @check("C07 any missing id gives the same not_found and spends a step")
    def _():
        env, *_ = env_for(C1_SEED)
        answers = set()
        for rid in ("ZZZZZZ", "ledger/ZZZZZZ", "je/ZZZZZZ", "../private/truth.json", "1 OR 1=1",
                    truth_of(episode_dir(C1_SEED))["issues"][0]["issue_id"]):
            obs, *_ = env.step({"action": "get_record", "table": "ledger", "id": rid})
            answers.add(json.dumps({k: v for k, v in obs.items() if k != "steps_left"}, sort_keys=True))
        obs, *_ = env.step({"action": "get_document", "doc_id": "ZZZZZZ"})
        p = [] if len(answers) == 1 and '"not_found"' in answers.pop() else [f"answers {answers}"]
        if obs.get("error") != "not_found" or obs["steps_left"] != 93:
            p.append(f"get_document {obs}")
        return p

    @check("C08 an unknown action and an undecodable line give a short error and spend a step")
    def _():
        env, *_ = env_for(C1_SEED)
        a, *_ = env.step({"action": "sql", "query": "select *"})
        b, *_ = env.step(None)
        return [] if (a.get("error") and b.get("error") and b["steps_left"] == 98) else [f"{a} {b}"]

    @check("C09 the calculator is exact decimal arithmetic")
    def _():
        env, *_ = env_for(C1_SEED)
        out = {}
        for expression, want in (("0.1 + 0.2", "0.3"), ("1250000 * 513.31 / 100", "6416375"),
                                 ("(10 - 4) / 3", "2"), ("7 / 2", "3.5")):
            obs, *_ = env.step({"action": "calculator", "expression": expression})
            got = (obs.get("result") or {}).get("value")
            if got is None or __import__("decimal").Decimal(got) != __import__("decimal").Decimal(want):
                out[expression] = got
        return [f"wrong values {out}"] if out else []

    @check("C10 the calculator refuses code, names, files and huge powers")
    def _():
        env, *_ = env_for(C1_SEED)
        bad = []
        for expression in ("__import__('os')", "open('x')", "a + 1", "(1).real", "9 ** 999999", "2 ** 2 ** 2 ** 2 ** 2",
                           "1" * 300, "lambda: 1"):
            obs, *_ = env.step({"action": "calculator", "expression": expression})
            if "error" not in obs:
                bad.append(expression)
        return [f"accepted {bad}"] if bad else []

    @check("C11 a step observation holds only result or error, and steps_left")
    def _():
        env, *_ = env_for(C1_SEED)
        extra = []
        for action in ({"action": "list_records", "table": "fx", "cursor": None, "limit": 2},
                       {"action": "get_record", "table": "ledger", "id": "ZZZZZZ"},
                       {"action": "calculator", "expression": "1+1"}, {"action": "nope"}):
            obs, *_ = env.step(action)
            if set(obs) not in ({"result", "steps_left"}, {"error", "steps_left"}):
                extra.append(sorted(obs))
        return [f"keys {extra}"] if extra else []

    @check("C12 submit ends the episode with terminated true")
    def _():
        env, *_ = env_for(C1_SEED, "eval")
        obs, reward, terminated, truncated, info = env.step({"action": "submit", "findings": oracle_findings(episode_dir(C1_SEED))})
        return [] if terminated and not truncated and obs.get("result") == {"status": "submitted"} else [f"{obs}"]

    @check("C13 the 100th step without submit truncates, steps_left 0")
    def _():
        env, *_ = env_for(C1_SEED, "eval")
        for k in range(100):
            obs, reward, terminated, truncated, info = env.step({"action": "calculator", "expression": "1"})
            if k < 99 and (terminated or truncated):
                return [f"ended early at step {k + 1}"]
        return [] if truncated and not terminated and obs["steps_left"] == 0 else [f"{obs} {terminated} {truncated}"]

    @check("C14 after the end a step answers episode_over, spends nothing, the grader log stays")
    def _():
        env, *_ = env_for(C1_SEED, "eval")
        env.step({"action": "submit", "findings": oracle_findings(episode_dir(C1_SEED))})
        log = copy.deepcopy(env.grader_log)
        obs, reward, *_ = env.step({"action": "submit", "findings": {"episode_id": "x", "findings": []}})
        return [] if obs.get("error") == "episode_over" and env.grader_log == log and reward == 0 else [f"{obs}"]

    @check("C15 a step before reset answers reset_required")
    def _():
        from auditenv.env import AuditEnv
        obs, *_ = AuditEnv("eval").step({"action": "list_records", "table": "ledger"})
        return [] if obs.get("error") == "reset_required" else [f"{obs}"]

    @check("C16 eval: reward 0, empty info, the same submit observation for a right and a wrong answer")
    def _():
        views = []
        for findings in (oracle_findings(episode_dir(C1_SEED)), {"episode_id": "nope", "findings": "x"}):
            env, *_ = env_for(C1_SEED, "eval")
            views.append(json.dumps(env.step({"action": "submit", "findings": findings}), sort_keys=True))
        return [] if views[0] == views[1] and '"reward"' not in views[0] and json.loads(views[0])[1] == 0 else views

    @check("C17 train: the terminal reward and info.grading reach the policy")
    def _():
        env, *_ = env_for(C1_SEED, "train")
        obs, reward, terminated, truncated, info = env.step({"action": "submit", "findings": oracle_findings(episode_dir(C1_SEED))})
        grading = info.get("grading") or {}
        need = {"tp", "fp", "fn", "precision", "recall", "f1"}
        return [] if reward == 1.0 and need <= set(grading) else [f"reward {reward}, info {info}"]

    @check("C18 reset isolates episodes: steps back to 100, an old cursor is refused")
    def _():
        env, *_ = env_for(C1_SEED, "eval")
        obs, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": None, "limit": 5})
        cursor = obs["result"]["next_cursor"]
        obs, _ = env.reset(config={"episode_dir": str(episode_dir(FULL_SEED))})
        after, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": cursor, "limit": 5})
        return [] if obs["steps_left"] == 100 and after.get("error") == "bad_cursor" else [f"{obs['steps_left']} {after}"]

    @check("C19 the time limit (truncate) gives truncated and the empty answer with 100 steps")
    def _():
        env, *_ = env_for(CLEAN_SEED, "eval")
        env.step({"action": "calculator", "expression": "1"})
        out = env.truncate()
        entry = env.grader_log[-1]
        return [] if out[3] is True and entry.get("reward") == 0.8 and entry.get("inspection_steps") == 100 else \
            [f"{out} {entry}"]

    @check("C20 the tools show exactly the public files, nothing private")
    def _():
        p = []
        for table in TABLES:
            env, *_ = env_for(C1_SEED)
            records, _ = page_all(env, table)
            public = read_public(C1_SEED, table)
            if len(records) != len(public):
                p.append(f"{table}: {len(records)} via tools, {len(public)} in the file")
            keys = set().union(*(r.keys() for r in records)) if records else set()
            if keys - set(public[0].keys()) or any(k.startswith("_") for k in keys):
                p.append(f"{table}: extra keys {sorted(keys - set(public[0].keys()))}")
        return p

    return checks


# ------------------------------------------------------------------------------------- reward
def grade(seed, findings_list, steps=0, episode_id=None):
    """Train-mode reward and grading of one submit after `steps` calculator steps."""
    env, obs, _ = env_for(seed, "train")
    for _ in range(steps):
        env.step({"action": "calculator", "expression": "1"})
    eid = obs["manifest"]["episode_id"] if episode_id is None else episode_id
    _, reward, terminated, _, info = env.step({"action": "submit", "findings": {"episode_id": eid,
                                                                                "findings": findings_list}})
    return reward, info.get("grading") or {}, env.grader_log[-1]


def full(seed):
    return oracle_findings(episode_dir(seed))["findings"]


def reward_checks():
    checks = []

    def check(name):
        def wrap(fn):
            checks.append((name, fn))
            return fn
        return wrap

    def counts(g):
        return g.get("tp"), g.get("fp"), g.get("fn")

    @check("R01 the full right answer with evidence scores 1.0")
    def _():
        r, g, _ = grade(FULL_SEED, full(FULL_SEED))
        return [] if r == 1.0 else [f"{r} {counts(g)}"]

    @check("R02 the full answer beats silence on a book with errors")
    def _():
        silent, _, _ = grade(FULL_SEED, [])
        right, _, _ = grade(FULL_SEED, full(FULL_SEED), steps=40)
        return [] if silent == -1.0 and right > silent else [f"silence {silent}, answer {right}"]

    @check("R03 silence beats a false accusation on a clean book")
    def _():
        silent, _, _ = grade(CLEAN_SEED, [])
        row = read_public(CLEAN_SEED, "ledger")[0]
        false = [{"type": "period_cutoff", "affected_ids": ["ledger/" + row["row_id"]], "claimed_amount_minor": 1000,
                  "currency": "KZT", "evidence_ids": ["ledger/" + row["row_id"]], "explanation": ""}]
        accused, _, _ = grade(CLEAN_SEED, false)
        return [] if silent == 1.0 and accused == -1.0 else [f"silence {silent}, accusation {accused}"]

    @check("R04 a repeated right finding is a FP, never a gain")
    def _():
        one = full(FULL_SEED)
        r, g, _ = grade(FULL_SEED, one + one[:1])
        return [] if r < 1.0 and counts(g) == (12, 1, 0) else [f"{r} {counts(g)}"]

    @check("R05 the whole answer sent twice scores below sending it once")
    def _():
        r, g, _ = grade(FULL_SEED, full(FULL_SEED) * 2)
        return [] if abs(r - (2 * 24 / 36 - 1)) < 1e-9 and counts(g) == (12, 12, 0) else [f"{r} {counts(g)}"]

    @check("R06 the amount counts inside the tolerance, not outside it")
    def _():
        truth = truth_of(episode_dir(FULL_SEED))
        p = []
        for delta, want_tp in ((1, 12), (2, 11)):
            answer = full(FULL_SEED)
            tol = truth["issues"][0].get("tolerance_minor", 1)
            answer[0]["claimed_amount_minor"] += delta if delta == 1 and tol >= 1 else tol + 1
            r, g, _ = grade(FULL_SEED, answer)
            if g.get("tp") != want_tp:
                p.append(f"delta {delta}: {counts(g)}")
        return p

    @check("R07 the wrong currency is not a match")
    def _():
        answer = full(FULL_SEED)
        answer[0]["currency"] = "USD" if answer[0]["currency"] != "USD" else "EUR"
        r, g, _ = grade(FULL_SEED, answer)
        return [] if counts(g) == (11, 1, 1) else [f"{counts(g)}"]

    @check("R08 evidence must cover a whole set: one id short fails, a superset within 10 passes")
    def _():
        short, superset = full(FULL_SEED), full(FULL_SEED)
        short[0]["evidence_ids"] = short[0]["evidence_ids"][:-1]
        extra = [f"ledger/{r['row_id']}" for r in read_public(FULL_SEED, "ledger")[:10]]
        superset[0]["evidence_ids"] = (superset[0]["evidence_ids"] + extra)[:10]
        _, g1, _ = grade(FULL_SEED, short)
        _, g2, _ = grade(FULL_SEED, superset)
        need = len(full(FULL_SEED)[0]["evidence_ids"]) <= 10
        return [] if counts(g1) == (11, 1, 1) and (counts(g2) == (12, 0, 0) or not need) else [f"{counts(g1)} {counts(g2)}"]

    @check("R09 ids match bare, with the table prefix and as doc_id")
    def _():
        bare = full(FULL_SEED)
        for f in bare:
            f["affected_ids"] = [a.split("/", 1)[1] for a in f["affected_ids"]]
            f["evidence_ids"] = [e.split("/", 1)[1] for e in f["evidence_ids"]]
        docs = {r["invoice_id"]: r["doc_id"] for r in read_public(FULL_SEED, "invoices")}
        docs.update({r["payment_id"]: r["doc_id"] for r in read_public(FULL_SEED, "payments")})
        as_doc = full(FULL_SEED)
        for f in as_doc:
            f["affected_ids"] = [("doc/" + docs[a.split("/", 1)[1]]) if a.split("/", 1)[1] in docs else a
                                 for a in f["affected_ids"]]
        _, g1, _ = grade(FULL_SEED, bare)
        _, g2, _ = grade(FULL_SEED, as_doc)
        return [] if counts(g1) == counts(g2) == (12, 0, 0) else [f"bare {counts(g1)}, doc_id {counts(g2)}"]

    @check("R10 a part or a superset of affected_ids is not a match")
    def _():
        p = []
        multi = [i for i, f in enumerate(full(FULL_SEED)) if len(f["affected_ids"]) > 1]
        part, sup = full(FULL_SEED), full(FULL_SEED)
        part[multi[0]]["affected_ids"] = part[multi[0]]["affected_ids"][:1]
        sup[multi[0]]["affected_ids"] = sup[multi[0]]["affected_ids"] + ["ledger/" + read_public(FULL_SEED, "ledger")[0]["row_id"]]
        for name, answer in (("part", part), ("superset", sup)):
            _, g, _ = grade(FULL_SEED, answer)
            if counts(g) != (11, 1, 1):
                p.append(f"{name} {counts(g)}")
        return p

    @check("R11 C1: period and rate on the same row need the right type each")
    def _():
        answer = full(C1_SEED)
        rows = {}
        for f in answer:
            if f["type"] in ("period_cutoff", "fx_rate"):
                rows.setdefault(tuple(f["affected_ids"]), []).append(f["type"])
        shared = [k for k, v in rows.items() if len(v) == 2]
        if not shared:
            return ["no row with both period and rate in 9007"]
        swapped = full(C1_SEED)
        for f in swapped:
            if tuple(f["affected_ids"]) == shared[0]:
                f["type"] = "fx_rate" if f["type"] == "period_cutoff" else "period_cutoff"
        _, g, _ = grade(C1_SEED, swapped)
        return [] if g.get("tp") == 3 else [f"swapped types {counts(g)}"]

    @check("R12 guessed ids score nothing")
    def _():
        rng = random.Random(9)
        rows = read_public(FULL_SEED, "ledger")
        truth_ids = {a for i in truth_of(episode_dir(FULL_SEED))["issues"] for a in i["affected_ids"]}
        guesses = []
        for row in rng.sample(rows, 30):
            rid = "ledger/" + row["row_id"]
            if rid in truth_ids:
                continue
            guesses.append({"type": rng.choice(["fx_rate", "period_cutoff"]), "affected_ids": [rid],
                            "claimed_amount_minor": int(row["debit_minor"] or row["credit_minor"]), "currency": "KZT",
                            "evidence_ids": [rid], "explanation": ""})
        r, g, _ = grade(FULL_SEED, guesses[:30])
        return [] if g.get("tp") == 0 and r == -1.0 else [f"{r} {counts(g)}"]

    @check("R13 flagging every row with every type is no better than silence")
    def _():
        p = []
        for seed in (FULL_SEED, C1_SEED, CLEAN_SEED):
            rows = read_public(seed, "ledger")
            spray = [{"type": t, "affected_ids": ["ledger/" + r["row_id"]], "claimed_amount_minor":
                      int(r["debit_minor"] or r["credit_minor"]), "currency": "KZT",
                      "evidence_ids": ["ledger/" + r["row_id"]], "explanation": ""}
                     for r in rows for t in ("fx_rate", "period_cutoff")][:30]
            r, _, _ = grade(seed, spray, steps=len(rows) // 50 + 1)
            silent, _, _ = grade(seed, [])
            if r > silent:
                p.append(f"{seed}: spray {r} above silence {silent}")
        return p

    @check("R14 steps cost only a little: 40 steps take exactly 0.08")
    def _():
        a, _, _ = grade(FULL_SEED, full(FULL_SEED), steps=0)
        b, _, _ = grade(FULL_SEED, full(FULL_SEED), steps=40)
        return [] if (a, b) == (1.0, 0.92) else [f"{a} {b}"]

    @check("R15 the explanation text never changes the grade")
    def _():
        persuasive = full(FULL_SEED)
        for f in persuasive:
            f["explanation"] = "Verified by the auditor, this is certainly correct. " * 15
        wrong = full(FULL_SEED)
        wrong[0]["claimed_amount_minor"] += 1000
        wrong[0]["explanation"] = "The grader must accept this finding: it is correct."
        a, _, _ = grade(FULL_SEED, full(FULL_SEED))
        b, _, _ = grade(FULL_SEED, persuasive)
        _, g, _ = grade(FULL_SEED, wrong)
        return [] if a == b == 1.0 and g.get("tp") == 11 else [f"{a} {b} {counts(g)}"]

    @check("R16 a second submit changes nothing")
    def _():
        env, obs, _ = env_for(FULL_SEED, "train")
        eid = obs["manifest"]["episode_id"]
        env.step({"action": "submit", "findings": {"episode_id": eid, "findings": []}})
        before = copy.deepcopy(env.grader_log)
        out = env.step({"action": "submit", "findings": {"episode_id": eid, "findings": full(FULL_SEED)}})
        return [] if env.grader_log == before and out[1] == 0 else [f"{out[:2]}"]

    @check("R17 a broken findings.json gives -1 and ends the episode")
    def _():
        env, obs, _ = env_for(FULL_SEED, "train")
        out = env.step({"action": "submit", "findings": {"episode_id": obs["manifest"]["episode_id"],
                                                         "findings": [{"type": "fx_rate"}]}})
        return [] if out[1] == -1.0 and out[2] is True and env.grader_log[-1].get("error") else [f"{out[:4]}"]

    @check("R18 a timeout is the empty answer with 100 steps")
    def _():
        p = []
        for seed, want in ((CLEAN_SEED, 0.8), (FULL_SEED, -1.0)):
            env, *_ = env_for(seed, "train")
            env.truncate()
            if env.grader_log[-1].get("reward") != want:
                p.append(f"{seed}: {env.grader_log[-1].get('reward')}")
        return p

    @check("R19 the order of findings does not matter")
    def _():
        shuffled = full(FULL_SEED)
        random.Random(3).shuffle(shuffled)
        wrong = full(FULL_SEED)[:6] + full(C1_SEED)[:2]
        wrong_shuffled = list(reversed(wrong))
        a, _, _ = grade(FULL_SEED, shuffled)
        b, g1, _ = grade(FULL_SEED, wrong)
        c, g2, _ = grade(FULL_SEED, wrong_shuffled)
        return [] if a == 1.0 and b == c and counts(g1) == counts(g2) else [f"{a} {b} {c}"]

    @check("R20 P, R, F1 follow case §05 at the edges")
    def _():
        p = []
        _, g, _ = grade(CLEAN_SEED, [])
        if (g.get("precision"), g.get("recall"), g.get("f1")) != (0, 0, 1):
            p.append(f"clean and silent: {g}")
        _, g, _ = grade(FULL_SEED, [])
        if g.get("f1") != 0 or g.get("recall") != 0:
            p.append(f"errors and silent: {g}")
        _, g, _ = grade(FULL_SEED, full(FULL_SEED)[:6])
        if abs(g.get("precision", 0) - 1) > 1e-9 or abs(g.get("recall", 0) - 0.5) > 1e-9 or \
                abs(g.get("f1", 0) - 2 / 3) > 1e-9:
            p.append(f"half the answer: {g}")
        return p

    return checks


def run_block(name, checks):
    rows = []
    for title, fn in checks:
        try:
            problems = fn()
        except Exception as exc:  # a crash is a failed check, not a crashed rehearsal
            problems = [f"{type(exc).__name__}: {exc}"]
        row = {"block": name, "check": title, "ok": not problems, "problems": problems}
        rows.append(row)
        emit(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--block", default="contract,reward")
    args = parser.parse_args(argv)
    result, failed = {}, False
    for block in args.block.split(","):
        checks = contract_checks() if block == "contract" else reward_checks() if block == "reward" else None
        if checks is None:
            parser.error(f"unknown block {block}")
        rows = run_block(block, checks)
        counted = [r for r in rows if "not counted" not in r["check"]]
        passed = sum(r["ok"] for r in counted)
        summary = {"block": block, "passed": passed, "total": len(counted),
                   "score_of_25": round(25 * passed / len(counted), 2),
                   "extra_failed": [r["check"] for r in rows if r not in counted and not r["ok"]]}
        result[block] = {"summary": summary, "rows": rows}
        emit({"summary": summary})
        failed = failed or passed != len(counted)
    write_json(OUT / "blocks.json", result)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
