# Golden episode v0.1

Hand-checkable fixture for the AI auditor RL environment. Company "Demo Logistics", base KZT, UTC+05:00, period 2026-09, seed 20260919.

Contents

- `build_golden.py` builds the episode into `<out>/public` and `<out>/private`
- `verify_golden.py` recomputes every truth amount from public files plus the evidence ids in truth.json only
- `episode/` is the built output

Run

```
python -I build_golden.py episode
python -I verify_golden.py episode      # expect 8/8 issues recomputed from evidence only
```

What it covers

- duplicate_invoice (S1, AP, EUR), payment_allocation (S2, AR, USD), fx_rate (S3, AR, USD), period_cutoff (S4, AR, KZT)
- coupled C1 (period + fx + allocation in one receipt) and C2 (duplicate bill paid on the copy)
- 8 correct twins listed in private/provenance.json

Known gaps, on purpose

- Small fixture. It does not meet volume invariants such as INV-03 (10 genuine rows on each side of the period)
- provenance.json has root causes but no `changes` block yet. The real generator derives `changes` from a clean versus injected diff
- Verified: byte-identical output under different PYTHONHASHSEED, no type names or private words in public files
