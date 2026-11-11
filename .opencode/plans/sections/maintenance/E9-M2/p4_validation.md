# P4 radius/bin initialization validation

P3 prerequisite: `378aaa2c0`, PR #1627; all required P3 checks passed before P4.
Native single-material radius and lognormal bin initializers now return owned
float64 ParticleData without constructing a facade. Kind is explicit; PDF grids
are retained and densities use fixed V=1. Direct multi-species mass input stays
with ParticleDataBuilder. Defaults and valid-input generator algorithms are
unchanged. Invalid GSD <= 1 is rejected explicitly.

Repository `.venv`, 2026-11-10 (America/Denver):

- Native initialization, data-builder, representation-builder and population
  tests: 82 passed (`pytest ... -q --no-cov`).
- Untargeted `.opencode/tools/run_pytest.py`, host device/cache access:
  6,844 passed, 19 skipped, 92% coverage; configured 80% gate passed.
- `.opencode/tools/run_linters.py`: Ruff and mypy passed.

The independent arithmetic tests distinguish PMF sums from PDF product
quadrature and cover zero PDFs, empty discrete inputs, fresh arrays, default
parameters, unit equivalence and V=.25/4 rejection. Converted non-SI reference
volume roundoff within four float64 ulps is canonicalized to exactly one;
direct SI volume admission stays exact. Concentration unit conversion is
compared with explicit relative tolerance, not bitwise equality.
Local logs: `/tmp/e9-m2-p4-{full,focused,lint}.log`.
