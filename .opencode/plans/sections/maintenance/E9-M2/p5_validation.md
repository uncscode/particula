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
