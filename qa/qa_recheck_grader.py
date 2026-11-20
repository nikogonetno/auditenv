"""QA re-check of B's grader (auditenv.grader) against the QA reference matcher.

1. E-24, all 23 cases of qa_reward.py, B versus the QA STRICT reference.
2. Differential fuzz, random submissions built from oracle findings, mutated copies,
   duplicates and junk. B and the reference must agree on TP, FP, FN and reward.
3. Order invariance (E-26) on every fuzz case.
The QA reference below is written from SPEC §5.2 and §5.4, not from matcher.py.

Usage: python -I qa_recheck_grader.py <repo_root> [n_fuzz]
"""
import copy
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
N_FUZZ = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
sys.path.insert(0, str(ROOT))
from auditenv.grader import score  # noqa: E402  (B's code under test)
from tests.golden import oracle_findings, public_record_ids, truth_v2  # noqa: E402

TYPES = ["duplicate_invoice", "payment_allocation", "fx_rate", "period_cutoff"]
PREFIXES = {"allocation", "counterparty", "doc", "fx", "invoice", "ledger", "line", "payment"}


# ------------------------------------------------------------------ QA reference, independent
def ref_norm(x):
    head, sep, rest = x.partition("/")
    return rest if sep and rest and head in PREFIXES else x


def ref_valid(f):
    if type(f) is not dict:
        return False
    t, a, m, c, e = (f.get(k) for k in ("type", "affected_ids", "amount_minor", "currency", "evidence"))
    return (type(t) is str and t in TYPES and type(a) is list and a and all(type(x) is str for x in a)
            and type(m) is int and m >= 0 and type(c) is str and len(c) == 3 and c.isascii() and c.isalpha()
            and c.isupper() and type(e) is list and len(e) <= 10 and all(type(x) is str for x in e))


def ref_ok(f, t):
    if not ref_valid(f) or f["type"] != t["type"]:
        return False
    if {ref_norm(x) for x in f["affected_ids"]} != {ref_norm(x) for x in t["affected_ids"]}:
        return False
    if abs(f["amount_minor"] - t["expected_amount_minor"]) > t.get("tolerance_minor", 1) or f["currency"] != t["currency"]:
        return False
    ev = {ref_norm(x) for x in f["evidence"]}
    return any({ref_norm(x) for x in s} <= ev for s in t["evidence_sets"])


def ref_score(F, truth, k):
    issues = truth["issues"]
    adj = [[j for j, f in enumerate(F) if ref_ok(f, t)] for t in issues]
    owner = {}

    def aug(i, seen):
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or aug(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    tp = sum(aug(i, set()) for i in range(len(issues)))
    fp, fn = len(F) - tp, len(issues) - tp
    f1 = Fraction(1) if 2 * tp + fp + fn == 0 else Fraction(2 * tp, 2 * tp + fp + fn)
    return tp, fp, fn, float(max(Fraction(-1), min(Fraction(1), 2 * f1 - 1 - Fraction(2, 1000) * k)))


def b_score(F, truth, k):
    r = score(copy.deepcopy(F), copy.deepcopy(truth), k)
    return r["tp"], r["fp"], r["fn"], r["reward"]


# ------------------------------------------------------------------ 1. the 23 cases
T = truth_v2()
CLEAN = {"clean": True, "issues": []}
O = oracle_findings(T)
ALL = public_record_ids()
BARE = [ref_norm(x) for x in ALL]
BASE = "KZT"


def edit(fs, **kw):
    out = []
    for f in fs:
        g = copy.deepcopy(f)
        for key, v in kw.items():
            g[key] = v(g) if callable(v) else v
        out.append(g)
    return out


flag_all = [{"type": ty, "affected_ids": [i], "amount_minor": 0, "currency": BASE, "evidence": [i]} for i in ALL for ty in TYPES]
spray = [{"type": ty, "affected_ids": list(ALL), "amount_minor": 0, "currency": BASE, "evidence": ALL[:10]} for ty in TYPES]
pairs_c1 = {}
for idx, t in enumerate(T["issues"]):
    pairs_c1.setdefault(tuple(t["affected_ids"]), []).append(idx)
c1_idx = {i for v in pairs_c1.values() if len(v) > 1 for i in v}
swap = {"fx_rate": "period_cutoff", "period_cutoff": "fx_rate"}
pad = lambda n: (lambda g: (g["evidence"] + [x for x in ALL if ref_norm(x) not in {ref_norm(y) for y in g["evidence"]}])[:n])

CASES = [
    ("oracle, 0 шагов", O, 0, T),
    ("oracle, 40 шагов", O, 40, T),
    ("молчание", [], 0, T),
    ("молчание, чистая компания", [], 0, CLEAN),
    ("тайм-аут, 100 шагов", [], 100, T),
    ("тайм-аут, чистая компания", [], 100, CLEAN),
    ("пометить всё, 40 шагов", flag_all, 40, T),
    ("пометить всё, чистая компания", flag_all, 40, CLEAN),
    ("1 ложный вывод на чистой", O[:1], 0, CLEAN),
    ("каждый вывод дважды", O + O, 0, T),
    ("сумма мимо на 2", edit(O, amount_minor=lambda g: g["amount_minor"] + 2), 0, T),
    ("не та валюта", edit(O, currency=lambda g: "USD" if g["currency"] != "USD" else "EUR"), 0, T),
    ("без доказательств", edit(O, evidence=[]), 0, T),
    ("нет одного ID в доказательствах", edit(O, evidence=lambda g: g["evidence"][1:]), 0, T),
    ("доказательства добиты до 10", edit(O, evidence=pad(10)), 0, T),
    ("11 ID в доказательствах", edit(O, evidence=pad(11)), 0, T),
    ("ID с префиксом таблицы", O, 0, T),
    ("ID без префикса", edit(O, affected_ids=lambda g: [ref_norm(x) for x in g["affected_ids"]],
                              evidence=lambda g: [ref_norm(x) for x in g["evidence"]]), 0, T),
    ("несуществующие ID", edit(O, affected_ids=["ZZZZZZ"], evidence=["ZZZZZZ"]), 0, T),
    ("распыление ID", spray, 0, T),
    ("лишний ID в affected_ids", edit(O, affected_ids=lambda g: g["affected_ids"] + [ALL[0]]), 0, T),
    ("половина oracle", O[: len(O) // 2], 0, T),
    ("oracle плюс 1 ложный", O + [{"type": "fx_rate", "affected_ids": [ALL[0]], "amount_minor": 5, "currency": BASE, "evidence": [ALL[0]]}], 0, T),
    ("C1, курс и период перепутаны", [dict(f, type=swap[f["type"]]) if i in c1_idx else f for i, f in enumerate(O)], 0, T),
]

print(f"truth issues {len(T['issues'])}, records {len(ALL)}")
bad = 0
for name, F, k, truth in CASES:
    b, r = b_score(F, truth, k), ref_score(F, truth, k)
    same = b == r
    bad += not same
    print(f"{'OK  ' if same else 'DIFF'} {name:34s} B {b[3]:+.3f} ({b[0]}/{b[1]}/{b[2]})   QA {r[3]:+.3f} ({r[0]}/{r[1]}/{r[2]})")
print(f"E-24: {len(CASES) - bad} of {len(CASES)} cases agree")

# ------------------------------------------------------------------ 2 and 3. differential fuzz + order invariance
rng = random.Random(20261119)


def mutate(f):
    g = copy.deepcopy(f)
    m = rng.randrange(12)
    if m == 0: g["amount_minor"] += rng.choice([-2, -1, 1, 2, 10**6])
    elif m == 1: g["currency"] = rng.choice(["USD", "EUR", "KZT", "kzt", "US", "USDD", 1])
    elif m == 2: g["evidence"] = g["evidence"][1:]
    elif m == 3: g["evidence"] = g["evidence"] + rng.sample(ALL, rng.randint(0, 8))
    elif m == 4: g["affected_ids"] = g["affected_ids"] + [rng.choice(ALL)]
    elif m == 5: g["type"] = rng.choice(TYPES + ["Fx_Rate", "", None])
    elif m == 6: g["affected_ids"] = [ref_norm(x) for x in g["affected_ids"]]
    elif m == 7: g["amount_minor"] = rng.choice([True, float(g["amount_minor"]), str(g["amount_minor"]), -1, None])
    elif m == 8: g.pop(rng.choice(list(g)))
    elif m == 9: g["evidence"] = g["evidence"] * rng.randint(2, 4)
    elif m == 10: g["affected_ids"] = rng.choice([[], "x", None, [1], g["affected_ids"] * 2])
    else: g["evidence"] = [("payment/" if rng.random() < .5 else "") + ref_norm(x) for x in g["evidence"]]
    return g


fuzz_bad, perm_bad = 0, 0
for n in range(N_FUZZ):
    truth = CLEAN if rng.random() < 0.2 else T
    F = []
    for _ in range(rng.randint(0, 14)):
        src = rng.choice(O)
        r = rng.random()
        F.append(copy.deepcopy(src) if r < 0.45 else mutate(src) if r < 0.85 else rng.choice([
            {"type": rng.choice(TYPES), "affected_ids": [rng.choice(ALL)], "amount_minor": rng.randint(0, 10**9),
             "currency": BASE, "evidence": rng.sample(ALL, 3)}, None, [], "x", 7, {}]))
    k = rng.randint(0, 120)
    b, r = b_score(F, truth, k), ref_score(F, truth, k)
    if b != r:
        fuzz_bad += 1
        if fuzz_bad <= 5:
            print("FUZZ DIFF", n, b, r, json.dumps(F, default=str)[:300])
    G = F[:]
    rng.shuffle(G)
    if b_score(G, truth, k) != b:
        perm_bad += 1
print(f"fuzz: {N_FUZZ - fuzz_bad} of {N_FUZZ} random submissions agree with the QA reference")
print(f"E-26: {N_FUZZ - perm_bad} of {N_FUZZ} shuffled submissions keep TP, FP, FN and reward")
sys.exit(1 if bad or fuzz_bad or perm_bad else 0)
