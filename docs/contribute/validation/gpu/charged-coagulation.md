# Charged coagulation and ion-source validation

Development evidence, 2026-11-13, branch
`codex/gpu-charged-coagulation-lucky-few`, based on `19a8fb11d`.
These results concern the implementation on this branch; no release is claimed.

## Implemented boundary

The five existing charged models can run directly or through prepared Warp
execution. A bounded parallel reduction supplies the exact pair maximum to the
existing stochastic sampler. Prepared CUDA capture and same-device checkpoint
continuation retain the selected model. Existing Brownian defaults remain.

The CPU Dyachkov dimensionless formula now evaluates differences with `expm1`
and an equivalent rearrangement to avoid cancellation near neutrality. The
analytic neutral limit is checked independently; CUDA agreement does not define
that limit. CUDA decimal fit coefficients are retained as explicit fp64
constants rather than rounding through float32 literals.

`IonPairSource` injects explicit positive/negative particles with unit counts,
independent persistent source randomness, a fixed-capacity rejection boundary,
and cumulative external-mass accounting. `IonNeutralCleanup` exports only
neutral particles made entirely from both reserved ion materials, with persistent
species mass and particle ledgers. Neither establishes ion collision physics or
the completed Lucky Few survival study.

The LF2013 helper rejects unsupported domains by default. Optional lower/upper
values for omitted repulsive fluxes are uncertainty bounds, not an extrapolated
fit or an atmospheric validation result. Same-sign singly charged seed particles
with diameter 3 nm fall below the corresponding fit's lower radius, approximately
5.308 nm. Production treatment remains a scientific decision for the full study.

## Device and checks

Host CUDA: NVIDIA GeForce RTX 5060, 8151 MiB; Warp 1.11.1; CUDA toolkit 12.9,
driver 13.4. The restricted sandbox cannot see the driver; CUDA qualification
was performed with host access. No CPU fallback was used for CUDA cases.

Commands below used `WARP_CACHE_PATH=/tmp/particula-charged-warp` where relevant.

| Check | Result |
| --- | --- |
| `.venv/bin/pytest particula/gpu/kernels/tests/coagulation_test.py particula/execution/tests/prepared_test.py particula/execution/tests/prepared_restart_capture_test.py -q --no-cov -Werror` | 942 passed, host CPU/CUDA |
| `.venv/bin/pytest particula/gpu/dynamics/tests/charged_models_test.py particula/execution/tests/prepared_charged_test.py particula/dynamics/ions/tests particula/dynamics/coagulation/tests/charged_dimensionless_kernel_test.py -q --no-cov -Werror` | 43 passed, host CPU/CUDA |
| `.venv/bin/pytest particula/execution/tests/prepared_ions_test.py particula/gpu/kernels/tests/ion_source_test.py particula/execution/tests/prepared_restart_capture_test.py particula/execution/tests/prepared_charged_test.py -q --no-cov -Werror` | 62 passed, host CPU/CUDA |
| `.venv/bin/pytest particula -Werror --cov=particula --cov-report=term-missing --cov-fail-under=80` | 10,047 passed, 14 skipped, 1 xfailed; 92.04% coverage; host CPU/CUDA |
| `.venv/bin/pytest examples_tests scripts/tests -q --no-cov -Werror` | 459 passed, 1 skipped; host execution |
| `.venv/bin/python scripts/docs_examples.py refresh gpu-charged-models` | Complete CUDA run; saved figure visually inspected |
| `.venv/bin/python -u scripts/docs_examples.py check` | Passed; all saved bundles current and CPU-policy examples executed; CUDA/manual policies require explicit refresh |
| `.venv/bin/ruff check particula/ examples_tests/ scripts/ --no-fix` | Passed |
| `.venv/bin/ruff format particula/ examples_tests/ scripts/ --check` | Passed |
| `.venv/bin/mypy particula/ --ignore-missing-imports` | Passed |
| `.venv/bin/python scripts/check_docs_structure.py` | 0 errors |
| `npm --prefix website test` | 3 passed |
| `npm --prefix website run build` | Passed; 0 link errors |

The full package run at `80e271627` includes paired-source integration,
composition and zero-production tests. Subsequent cleanup changes are covered
by the focused checks below; that earlier full run does not qualify them.
The existing functional and object charged examples
were also refreshed; all four saved figures were visually inspected. Notebook
execution required host access; restricted notebook processes were stopped
after stalling during kernel startup. A restricted tooling run also hit
the read-only Warp cache; the successful host rerun above supersedes it.

Hosted CI is separate from these local results and remains pending on the draft PR.
No speedup or atmospheric survival probability is established by this record.

## Neutral-product cleanup follow-up

Cleanup tests check that neutral aerosols remain, charged ion-only particles
remain, each exported species is accounted for, repeated cleanup is a no-op,
and ledger/state continuation survives checkpoint and native CUDA capture.
Invalid reserved indices reject at preparation; evolving fractional counts
fault before modifying the affected box. The source and cleanup retain distinct
families so the example must explicitly choose their ordering.

Host command: `WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest
particula/execution/tests/prepared_ion_cleanup_test.py
particula/execution/tests/prepared_ions_test.py
particula/execution/tests/prepared_restart_capture_test.py -q --no-cov -Werror`:
58 passed, including actual CUDA. Ruff and mypy passed (571 source files).

## Original-seed lineage follow-up

`SeedCohort` stores unresolved original-lineage counts and persistent background
labels separately from particle inventories. It consumes only applied compacted
collision pairs and observes size after each physical process. Outcome counts
partition the original cohort into reached, captured and unresolved lineages.

The affected prepared suite passed 171 tests on host CPU/CUDA:

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest \
  particula/execution/tests/prepared_cohort_test.py \
  particula/execution/tests/prepared_test.py \
  particula/execution/tests/prepared_charged_test.py \
  particula/execution/tests/prepared_ions_test.py \
  particula/execution/tests/prepared_ion_cleanup_test.py \
  particula/execution/tests/prepared_restart_capture_test.py \
  -q --no-cov -Werror
```

Tests cover two seed lineages in one merged particle, merger-induced threshold
crossing, capture precedence, irrevocable previous success, unresolved censoring,
real condensation growth, actual CUDA capture, restart before/after merging,
corrupt checkpoint outcome rejection and conflicting ion material maps.
Ruff, formatting and mypy passed (575 source files). These tests establish
bookkeeping and integration; survival-probability convergence is still pending.

## Ion attachment numerics and condensation exclusion

The LF2013 device helper evaluates the published degree-23 polynomials in a
centered logarithmic-radius basis. Coefficients are transformed once using
70-digit decimal arithmetic and uploaded as fp64 values. This is an algebraic
basis change, not a new fit or interpolation. Tests independently evaluate the
original decimal coefficients at nine radii in each valid interval, covering
all 201 aerosol charges and both ion polarities. CPU and CUDA agree with those
references within 1e-10 relative tolerance. Default unsupported rates return a
fault sentinel; optional lower/upper bounds remain explicit. Device helpers
classify free ions, inert neutral products, aerosols and unsupported ion-only
clusters. They are not yet connected to the atmospheric collision scheduler.

`IonCondensationExclusion` gates ion-only proposals before gas-inventory
limiting in every condensation substep. Tests verify exact ion/product masses,
continued uptake by an aerosol carrying ion mass, particle/gas conservation,
isothermal and latent-heat capture/restart, and configuration rejection.
Mixed physical material densities use the CPU surface model's constant Kelvin
density rather than replacing physical densities; a native CPU comparison
passes at 2e-12 relative tolerance. The former mixed-material rejection test
now exercises the still-unsupported nonuniform surface-density case, while
the new parity test covers the newly supported material-density case.

Host command (171 passed, actual CPU/CUDA):

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest \
  particula/execution/tests/prepared_ion_condensation_test.py \
  particula/execution/tests/prepared_test.py \
  particula/execution/tests/prepared_cohort_test.py \
  particula/execution/tests/prepared_restart_capture_test.py \
  particula/gpu/dynamics/tests/ion_attachment_test.py \
  particula/dynamics/ions/tests/attachment_test.py \
  -q --no-cov -Werror
```

The direct condensation regression, parallel reductions and autodiff were also
checked after the shared proposal signature changed:

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest \
  particula/gpu/kernels/tests/condensation_test.py \
  particula/gpu/kernels/tests/condensation_autodiff_test.py \
  particula/gpu/kernels/tests/condensation_parallel_test.py \
  -q --no-cov -Werror
```

Result: 324 passed, 3 skipped. Ruff, formatting and mypy (581 source files)
passed; the website built with zero link errors.
