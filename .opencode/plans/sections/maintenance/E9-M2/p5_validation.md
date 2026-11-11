# P5 resolved initialization validation

P4 prerequisite: `e24cb7990`, PR #1628; P4 validation passed before P5.
Direct resolved initialization supports explicit mass/species arrays, per-box
volume, fractional counts, and explicitly identified density input converted
once to counts. Sampled lognormal initialization preserves existing defaults,
unit count weights, and the NumPy/SciPy random-state path. Deterministic input
rejection precedes sampling. Returned arrays are owned and writable.

Repository `.venv`, 2026-11-10 (America/Denver):

- Initialization/data-builder/legacy-builder/population checks: 118 passed.
- Untargeted `.opencode/tools/run_pytest.py`, host device/cache access:
  6,880 passed, 19 skipped, 92% coverage; configured 80% gate passed.
- `.opencode/tools/run_linters.py`: Ruff and mypy passed.

`resolved_initialization_test.py` independently verifies each species inventory
and density at V=.25/1/4 for count, m^-3 and cm^-3 inputs; multibox SI conversion;
scalar/singleton/redundant forms; owned broadcasts; empty/zero populations;
lossy shape rejection with unchanged inputs; controlled samples; real default
20,000-sample behavior; and unchanged global RNG routing. It maps retained
resolved-builder assertions to native behavior without reproducing lossy legacy
first-row/column truncation. Canonical (B,N) charge shape has explicit priority.
Local logs: `/tmp/e9-m2-p5-{full,focused,lint}.log`.

## Main update and base-install repair (2026-11-11)

Merged main at `b1e6198aa`, retaining P4's optional-Pint and minimum-grid fixes
alongside P5's resolved capabilities. The capability-ledger conflict preserves
both the P4 grid contract and P5 resolved evidence.

Resolved volume and canonical number-density inputs now use the same identity
unit-factor path as P4. Counts remain unchanged; density still multiplies by
physical volume exactly once. Non-SI inputs retain optional Pint conversion.
Regression tests force an absent unit registry for count/density inventory and
controlled sampling at volumes 0.25, 1 and 4, and verify missing non-SI
conversions reject without mutation or random sampling.

Repository `.venv` focused validation across both initialization modules:
**84 passed** with Pint; **77 passed, 7 skipped** with Pint imports blocked
before package import. Only non-SI conversion cases skip. Exact CI Ruff scope
and format check pass (**532 files formatted**); mypy passes **509 files**.
