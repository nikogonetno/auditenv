"""Probe policies of the jury rehearsal (docs/tasks/V_rehearsal.md, part A items 6 and 7).

    python3 qa/rehearsal/probe_policy.py MODE

Standalone on purpose: it speaks only the JSON-lines protocol of the README (one runner message per
line in, one action per line out) and imports nothing from the repository, as an external policy would.

Limit modes (item 6):
  steps            list_records forever, never submit
  submit_at:N      N-1 list_records, then an empty valid submit as step N
  sleep            read the first message and sleep
  close            close stdout at once
  repeat_submit    a valid empty submit, then a second one
  bad:<case>       one submit whose structure breaks case §05 (see BAD below)
  edge:<case>      one submit exactly at a limit of case §05, structurally valid
Hostile mode (item 7):
  hostile          tries files, /proc, the environment and the runner, then malformed lines and
                   actions, then a broken submit. What it could reach goes out as an action
                   {"action": "probe_report", ...}, so the trajectory carries it to the harness.
"""
import glob
import hashlib
import json
import os
import signal
import sys
import time


def read_message():
    line = sys.stdin.buffer.readline()
    return json.loads(line.decode("utf-8")) if line else None


def send_raw(data):
    sys.stdout.buffer.write(data if data.endswith(b"\n") else data + b"\n")
    sys.stdout.buffer.flush()


def send(action):
    send_raw(json.dumps(action, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def over(message):
    return message is None or message.get("terminated") or message.get("truncated")


def finding(n=1, evidence=1, explanation=1, amount=1):
    return {"type": "duplicate_invoice", "affected_ids": [f"invoice/PROBE{n % 10}"], "claimed_amount_minor": amount,
            "currency": "KZT", "evidence_ids": [f"invoice/E{i:05d}" for i in range(evidence)],
            "explanation": "x" * explanation}


def answer(episode_id, findings):
    return {"action": "submit", "findings": {"episode_id": episode_id, "findings": findings}}


BAD = {
    "findings_not_list": lambda eid: {"episode_id": eid, "findings": "all of them"},
    "no_episode_id": lambda eid: {"findings": []},
    "wrong_episode_id": lambda eid: {"episode_id": "ZZZZZZ", "findings": []},
    "amount_float": lambda eid: {"episode_id": eid, "findings": [finding(amount=1.0)]},
    "amount_bool": lambda eid: {"episode_id": eid, "findings": [finding(amount=True)]},
    "amount_negative": lambda eid: {"episode_id": eid, "findings": [finding(amount=-5)]},
    "amount_string": lambda eid: {"episode_id": eid, "findings": [finding(amount="100")]},
    "currency_lower": lambda eid: {"episode_id": eid, "findings": [dict(finding(), currency="kzt")]},
    "type_unknown": lambda eid: {"episode_id": eid, "findings": [dict(finding(), type="fraud")]},
    "affected_empty": lambda eid: {"episode_id": eid, "findings": [dict(finding(), affected_ids=[])]},
    "evidence_not_list": lambda eid: {"episode_id": eid, "findings": [dict(finding(), evidence_ids="invoice/X")]},
    "findings_31": lambda eid: {"episode_id": eid, "findings": [finding(n) for n in range(31)]},
    "evidence_11": lambda eid: {"episode_id": eid, "findings": [finding(evidence=11)]},
    "explanation_1001": lambda eid: {"episode_id": eid, "findings": [finding(explanation=1001)]},
    "findings_object": lambda eid: "not an object",
}
EDGE = {
    "findings_30": lambda eid: {"episode_id": eid, "findings": [finding(n) for n in range(30)]},
    "evidence_10": lambda eid: {"episode_id": eid, "findings": [finding(evidence=10)]},
    "explanation_1000": lambda eid: {"episode_id": eid, "findings": [finding(explanation=1000)]},
    "empty": lambda eid: {"episode_id": eid, "findings": []},
}


# ---------------------------------------------------------------------------------- hostile
def try_read(path, limit=1 << 20):
    try:
        with open(path, "rb") as f:
            data = f.read(limit)
    except FileNotFoundError:
        return {"path": path, "result": "absent"}
    except OSError as exc:
        return {"path": path, "result": "denied", "errno": exc.errno}
    text = data.decode("utf-8", "replace")
    return {"path": path, "result": "readable", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()[:16],
            "looks_private": any(w in text for w in ("issue_id", "evidence_sets", "injections", "twin_id")),
            "has_seed_flag": "--seed" in text or "AUDITENV" in text, "head": text[:160]}


def probe_environment():
    report = {"pid": os.getpid(), "ppid": os.getppid(), "uid": os.getuid(), "cwd": os.getcwd(),
              "environ": {k: v[:120] for k, v in sorted(os.environ.items())}}
    ppid = os.getppid()
    report["self_environ"] = try_read("/proc/self/environ")
    report["parent_environ"] = try_read(f"/proc/{ppid}/environ")
    report["parent_cmdline"] = try_read(f"/proc/{ppid}/cmdline")
    try:
        parent_cwd = os.readlink(f"/proc/{ppid}/cwd")
        report["parent_cwd"] = parent_cwd
    except OSError as exc:
        parent_cwd, report["parent_cwd"] = None, f"denied errno {exc.errno}"
    fds = []
    try:
        for fd in sorted(os.listdir(f"/proc/{ppid}/fd"), key=int)[:64]:
            try:
                target = os.readlink(f"/proc/{ppid}/fd/{fd}")
            except OSError as exc:
                fds.append({"fd": fd, "result": f"denied errno {exc.errno}"})
                continue
            entry = {"fd": fd, "target": target}
            if target.startswith("/") and not target.startswith(("/dev/", "/proc/")):
                entry.update(try_read(f"/proc/{ppid}/fd/{fd}", 4096))
            fds.append(entry)
    except OSError as exc:
        fds.append({"result": f"denied errno {exc.errno}"})
    report["parent_fds"] = fds
    cmdlines = []
    for path in sorted(glob.glob("/proc/[0-9]*/cmdline"))[:400]:
        r = try_read(path, 4096)
        if r["result"] == "readable" and ("--seed" in r["head"] or "--config" in r["head"]):
            cmdlines.append(r)
    report["cmdlines_with_seed_or_config"] = cmdlines
    candidates = ["../private/truth.json", "../../private/truth.json", "private/truth.json",
                  "/app/tests/fixtures/mini_v03/private/truth.json"]
    candidates += sorted(glob.glob("/tmp/auditenv-episode-*/private/*"))[:10]
    for base in sorted({"/app", parent_cwd or "/app"}):
        for pattern in ("*/private/*.json", "*/*/private/*.json", "*/*/*/private/*.json",
                        "*/*/*/*/private/*.json", "runs/*", "runs/*/*"):
            candidates += sorted(glob.glob(os.path.join(base, pattern)))[:20]
    seen, reads = set(), []
    for path in candidates:
        if path not in seen and not os.path.isdir(path):
            seen.add(path)
            reads.append(try_read(path, 1 << 16))
    report["file_reads"] = reads
    target = os.path.join(parent_cwd or "/app", "probe_was_here.txt")
    try:
        with open(target, "x") as f:
            f.write("x")
        os.unlink(target)
        report["write_repo"] = "written"
    except OSError as exc:
        report["write_repo"] = f"denied errno {exc.errno}"
    try:
        os.kill(ppid, 0)
        report["signal_runner"] = "allowed"
    except OSError as exc:
        report["signal_runner"] = f"denied errno {exc.errno}"
    return report


RAW_LINES = [
    b"{not json",
    b"a" * (2 * 1024 * 1024),
    b"[" * 2000 + b"]" * 2000,
    b'{"action": "calculator", "expression": NaN}',
    b'{"action": "list_records", "table": "ledger", "limit": 1e999}',
    b'{"action": "list_records", "table": "ledger", "limit": ' + b"9" * 40 + b"}",
    b"\xff\xfe\xfd",
    b'{"action": "get_record", "table": "ledger", "id": "\\ud800"}',
    b"",
]


def hostile_actions(eid, cursor):
    other = cursor.replace(".ledger.", ".invoices.") if cursor else "c1.X.invoices.50"
    yield {"action": "read_file", "path": "../private/truth.json"}
    yield {"action": "shell", "command": "cat ../private/truth.json"}
    for table in ("truth", "private", "provenance", "../private/truth", "grader_log", "Ledger"):
        yield {"action": "list_records", "table": table, "cursor": None, "limit": 5}
    for bad_cursor in ("c1.ZZZZZZ.ledger.0", other, f"c1.{eid}.ledger.-50", f"c1.{eid}.ledger.999999999",
                       "../private", 5, ["c1"]):
        yield {"action": "list_records", "table": "ledger", "cursor": bad_cursor, "limit": 5}
    for limit in (0, 51, -1, "50", 50.0, True):
        yield {"action": "list_records", "table": "ledger", "cursor": None, "limit": limit}
    for rid in ("ZZZZZZ", "ledger/ZZZZZZ", "../private/truth.json", "je/ZZZZZZ", "1 OR 1=1", "truth/X",
                "private/truth.json", "issue/ZZZZZZ", "TW-01", ""):
        yield {"action": "get_record", "table": "ledger", "id": rid}
    yield {"action": "get_record", "table": "truth", "id": "ZZZZZZ"}
    yield {"action": "get_record", "table": "provenance", "id": "seed"}
    for doc in ("ZZZZZZ", "../../private/provenance.json", "doc/ZZZZZZ"):
        yield {"action": "get_document", "doc_id": doc}
    for expression in ("__import__('os').system('id')", "open('/etc/passwd')", "2**100000", "9" * 500, "1/0",
                       "(1" * 300 + ")" * 300, "seed", "1e400"):
        yield {"action": "calculator", "expression": expression}
    yield {"action": "submit", "findings": "everything"}


def hostile():
    message = read_message()
    if over(message):
        return
    infos, rewards = [message.get("info")], [0]
    observation = message.get("observation") or {}
    eid = (observation.get("manifest") or {}).get("episode_id", "")
    report = probe_environment()
    send({"action": "probe_report", "report": report})
    message = read_message()
    for raw in RAW_LINES:
        if over(message):
            return
        infos.append(message.get("info"))
        send_raw(raw)
        message = read_message()
    if over(message):
        return
    send({"action": "list_records", "table": "ledger", "cursor": None, "limit": 1})
    message = read_message()
    cursor = ((message or {}).get("observation") or {}).get("result", {}).get("next_cursor")
    for action in hostile_actions(eid, cursor):
        if over(message):
            return
        infos.append(message.get("info"))
        rewards.append(message.get("reward"))
        if action["action"] == "submit":
            send({"action": "probe_seen", "infos": sorted({json.dumps(i, sort_keys=True) for i in infos}),
                  "rewards": sorted({str(r) for r in rewards})})
            message = read_message()
            if over(message):
                return
        send(action)
        message = read_message()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "steps"
    if mode == "hostile":
        return hostile()
    message = read_message()
    if over(message):
        return
    eid = ((message.get("observation") or {}).get("manifest") or {}).get("episode_id")
    if mode == "sleep":
        time.sleep(3600)
        return
    if mode == "close":
        os.close(1)                  # sys.stdout.close() would leave fd 1 open (closefd=False)
        time.sleep(5)
        return
    if mode.startswith("bad:") or mode.startswith("edge:"):
        kind, case = mode.split(":", 1)
        send({"action": "submit", "findings": (BAD if kind == "bad" else EDGE)[case](eid)})
        read_message()
        return
    if mode == "repeat_submit":
        send(answer(eid, []))
        read_message()
        send(answer(eid, [finding()]))
        read_message()
        return
    submit_at = int(mode.split(":", 1)[1]) if mode.startswith("submit_at:") else None
    step = 0
    while not over(message):
        step += 1
        if submit_at is not None and step == submit_at:
            send(answer(eid, []))
        else:
            send({"action": "list_records", "table": "ledger", "cursor": None, "limit": 1})
        message = read_message()


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
