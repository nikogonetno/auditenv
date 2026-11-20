"""QA attacks on B's grader that the differential fuzz does not cover.

Reproduces D-1, D-2 and R-08 from B's own review, then probes input handling,
side effects, truth guards and cost. Prints facts only, every line is a measurement.
Usage: python -I qa_attack_grader.py <repo_root>
"""
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from auditenv.grader import parse_truth, score  # noqa: E402
from tests.golden import oracle_findings, read_csv, truth_v2  # noqa: E402

T = truth_v2()
O = oracle_findings(T)
CLEAN = {"clean": True, "issues": []}


def line(tag, text):
    print(f"{tag:6s} {text}")


# ---------------------------------------------------------------- D-1 str subclass in type
class AnyType(str):
    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash("fx_rate")


g = copy.deepcopy(O[0])
wrong = "fx_rate" if g["type"] != "fx_rate" else "period_cutoff"
g["type"] = AnyType(wrong)
r = score([g], {"clean": False, "issues": [T["issues"][0]]}, 0)
line("D-1", f"type is a str subclass claiming {wrong!r} for a {T['issues'][0]['type']!r} issue, TP {r['tp']}, reward {r['reward']}")

# ---------------------------------------------------------------- D-2 ledger/je/ slips through
t2 = copy.deepcopy(T)
t2["issues"][0]["affected_ids"] = ["ledger/je/A12BT6"]
try:
    parse_truth(t2)
    line("D-2", "truth with ledger/je/A12BT6 is ACCEPTED by parse_truth")
except ValueError as e:
    line("D-2", f"rejected: {e}")

# ---------------------------------------------------------------- truth guard, duplicate (type, affected)
t3 = copy.deepcopy(T)
twin = copy.deepcopy(t3["issues"][0])
twin["issue_id"] = "ZZZZZZ"
t3["issues"].append(twin)
try:
    parse_truth(t3)
    r = score(O, t3, 0)
    line("NEW-1", f"truth with two issues of equal type and affected_ids is ACCEPTED, oracle gets TP {r['tp']} FN {r['fn']} reward {r['reward']:.3f}")
except ValueError as e:
    line("NEW-1", f"rejected: {e}")

# ---------------------------------------------------------------- side effects on inputs
F0, T0 = copy.deepcopy(O + [{"x": 1}]), copy.deepcopy(T)
F1, T1 = copy.deepcopy(F0), copy.deepcopy(T0)
score(F1, T1, 3)
line("SIDE", f"score() leaves findings unchanged {F1 == F0}, truth unchanged {T1 == T0}")

# ---------------------------------------------------------------- R-08 period spray
led = read_csv("ledger.csv")
carriers = [r for r in led if not r["settles_invoice_id"] and (
    (r["source_type"] == "invoice" and r["account"] in ("1210", "3310")) or
    (r["source_type"] == "payment" and r["account"] == "1030"))]
spray = [{"type": "period_cutoff", "affected_ids": [f"ledger/{r['row_id']}"],
          "claimed_amount_minor": int(r["debit_minor"]) or int(r["credit_minor"]), "currency": "KZT",
          "evidence_ids": [f"{r['source_type']}/{r['source_id']}", f"ledger/{r['row_id']}"]} for r in carriers]
pages = -(-len(led) // 50)
for k in (0, pages):
    r = score(spray, T, k)
    line("R-08", f"period spray on {len(carriers)} carrier rows, {k} steps, TP {r['tp']} FP {r['fp']} FN {r['fn']}, reward {r['reward']:.3f}")
rc = score(spray, CLEAN, pages)["reward"]
rs = score(spray, T, pages)["reward"]
for c in (0.0, 0.12, 0.25):
    spray_avg = (1 - c) * rs + c * rc
    silence_avg = (1 - c) * -1 + c * 1
    line("R-08", f"clean share {c:.2f}, mean reward spray {spray_avg:+.3f} silence {silence_avg:+.3f}")
# break-even clean share where silence starts to win on average
line("R-08", f"silence wins on average once clean share > {(rs + 1) / (rs + 3):.3f}")

# ---------------------------------------------------------------- JSON level hazards for the env (not the grader)
for name, text in [("NaN amount", '{"claimed_amount_minor": NaN}'), ("Infinity amount", '{"claimed_amount_minor": Infinity}')]:
    v = json.loads(text)["claimed_amount_minor"]
    f = dict(O[0], claimed_amount_minor=v)
    line("JSON", f"{name}: json.loads accepts it as {type(v).__name__}, grader counts it invalid {score([f], T, 0)['tp'] == 0}")
try:
    json.loads('{"claimed_amount_minor": 1' + "0" * 5000 + "}")
    line("JSON", "int with 5001 digits parsed")
except ValueError as e:
    line("JSON", f"int with 5001 digits makes json.loads raise ValueError on this Python ({str(e)[:60]})")
try:
    json.loads("[" * 200000 + "]" * 200000)
    line("JSON", "200000 nested lists parsed")
except RecursionError:
    line("JSON", "200000 nested lists make json.loads raise RecursionError")

# ---------------------------------------------------------------- cost
for n, label in [(100_000, "copies of the oracle"), (100_000, "junk findings")]:
    F = (O * (n // len(O))) if "oracle" in label else [{"type": "fx_rate", "affected_ids": [str(i)], "claimed_amount_minor": i,
                                                       "currency": "KZT", "evidence_ids": [str(i)]} for i in range(n)]
    t0 = time.perf_counter()
    r = score(F, T, 0)
    line("COST", f"{len(F)} {label}: {time.perf_counter() - t0:.2f} s, TP {r['tp']} FP {r['fp']}")
big = [dict(O[0], evidence_ids=["x"] * 1_000_000)]
t0 = time.perf_counter()
score(big, T, 0)
line("COST", f"one finding with 1 000 000 evidence ids: {time.perf_counter() - t0:.2f} s")
