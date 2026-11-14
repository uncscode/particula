# Charged coagulation and atmospheric ion validation

The historical results below predate the environmental extension's charged
Knudsen normalization correction. They are evidence for their recorded
revisions, not current numerical qualification. A replacement confirmation
and publication refresh are required before treating the corrected results
as final. See [environmental validation](environment-sweep.md).

Development evidence, 2026-11-13, branch
`codex/gpu-charged-coagulation-lucky-few`, based on `19a8fb11d`.
These results concern the implementation on this branch; no release is claimed.
Sections retain chronological checkpoints. The latest charged numerical results
are in [completed qualification](#completed-charged-survival-qualification);
earlier pending-status notes describe their original checkpoints.
The [complete publication run](#complete-lucky-few-publication-run) records
the saved results for the [Lucky Few example](../../../examples/gpu/prepared/gpu_lucky_few.py).

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
values for omitted repulsive fluxes are explicit coefficient assumptions, not
bounds on survival probability or an extrapolated fit. Same-sign charged seeds
with diameter 3 nm fall below the corresponding fit's lower radius, approximately
5.308 nm. On 2026-11-14 the user selected both coefficient assumptions for
the survival sensitivity demonstration; neither is a fitted best estimate.

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

Hosted CI is separate from these local results. All six workflows passed at
`8d8f476fd`; newer increments require their own hosted results.
No CPU speedup or field-validated atmospheric survival prediction is established
by this record.

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

## Atmospheric collision integration and ion balance

The follow-up to `8d8f476fd` routes ion pairs through the existing sampler and
merge application. Both the exact pair maximum and acceptance probability use
the same derived ion classification and cached LF2013 rates. Validation faults
before collision RNG or physical merge mutation. Neutral products are inert;
same-sign ions do not cluster. Source, cleanup, condensation exclusion and cohort
material contracts are checked together. Derived scratch is recomputed at each
step and does not introduce a new checkpoint payload.

Host CPU/CUDA evidence, 2026-11-14 UTC:

| Command | Result |
| --- | --- |
| `.venv/bin/pytest particula/execution/tests particula/gpu/kernels/tests/brownian_parallel_test.py particula/gpu/dynamics/tests/ion_attachment_test.py -q --no-cov` | 1,721 passed before the final three composition/marker tests |
| `.venv/bin/pytest --cov=particula --cov-report=term-missing --cov-fail-under=80 -Werror` | 10,110 passed, 14 skipped, 1 xfailed; 91.85% coverage; includes all 30 atmospheric tests and actual CUDA |
| `.venv/bin/ruff check particula examples_tests scripts --no-fix` | Passed |
| `.venv/bin/ruff format particula examples_tests scripts --check` | 643 files formatted |
| `.venv/bin/mypy particula --ignore-missing-imports` | Passed, 586 files |
| `.venv/bin/python scripts/docs_examples.py refresh gpu-ion-balance` | Complete RTX 5060 run; figure visually inspected |
| `.venv/bin/python scripts/check_docs_structure.py` | Passed |
| `npm --prefix website test` | 3 passed |
| `npm --prefix website run build` | 453 pages; passed |

The ion-balance example simulates 64 independent boxes per q=0/2/10 condition,
256 slots per box, timestep 1 s and duration 2000 s. A separate sparse master
equation solution supplies the exact finite-population mean and variance;
its truncated upper-state probability is checked below 1e-10. Maximum observed
mean discrepancies were 1.58, 2.67 and 2.54 standard errors, below the fixed six
standard-error acceptance criterion. Net charge was exactly zero. Per-species
mass budgets included source input and neutral-product export, with maximum
relative residual below 1e-14. Observed peak occupied slots were 40, 70 and 128.
The current Python download matches the maintained source and compiles.

This qualifies the source/recombination example, not a survival conclusion.
The 3–50 nm study and its timestep/cleanup/capacity/ensemble convergence remain
outstanding. The production treatment of omitted LF2013 repulsive coefficients
is still awaiting the user's scientific choice; strict rejection is the default.

## Periodic cleanup and documentation reachability follow-up

Hosted documentation CI at `0288c548b` detected that the new ion-balance page
was not linked from authored navigation. The earlier local check ran before
that file was tracked and did not inventory it. Commit `fc18f012d` adds the
owning GPU index and guide links; the committed-file structure check reports
zero errors and its 13 tests pass. This corrects the earlier local-only
structure result rather than treating the failed hosted build as successful.

`IonNeutralCleanup.interval_steps` makes refresh frequency independent of the
physical timestep. A bounded per-box phase counter is checkpointed; cleanup
runs on multiples of the configured interval. Inert neutral products continue
to occupy slots between exports. Unit-count checks still run every call and
invalid checkpoint phases or changed intervals reject. CPU/CUDA cleanup,
atmospheric, cohort and restart regression passed 104 tests, including
midperiod capture/restart and default every-step behavior. Ruff and mypy passed.
The earlier full-package coverage result predates this periodic extension.


The subsequent combined study-helper/ion-exclusion/cleanup/atmospheric/cohort/
restart suite passed 125 tests on host CPU/CUDA. This caught and corrected a
shared-plan assumption: condensation exclusion must not inherit the cleanup
interval parameter. The complete ion-balance example was refreshed again and
its default every-step results retained.

A preliminary neutral survival pilot used the explicit defaults in
`scripts/gpu_studies/lucky_few.py`: 64 independent boxes, one 3 nm seed each,
32 initial 100 nm background particles per box, pulsed vapor replenishment,
and 20,000 s duration. It recorded 2 reached, 62 captured and 0 unresolved
lineages, with maximum mass residual 4.67e-15. The exact 95% success interval
is broad (0.003807–0.108371); this is not a converged production estimate.
The pending timestep/capacity/ensemble work and ion-domain decision remain in
the plan. No ion-effect conclusion follows from this q=0 pilot.


## Neutral growth and survival uncertainty controls

Host CUDA checks at library revision `5aa5b47ff` used the frozen physical
parameters in the survival helper. With background particles disabled, four
seeds per run all reached 50 nm at each of dt = 2, 1 and 0.5 s. First passage
was first observed at 1390 s in all three runs, using 10 s diagnostic spacing.
This satisfies the predeclared crossing-time spread criterion of the larger
of 10 s and 1% of the finest-step crossing time. It does not resolve passage
time within a diagnostic interval. Maximum mass residuals were 2.45e-15,
2.23e-15 and 3.45e-15, respectively.

Survival timestep qualification uses an absolute probability tolerance of
0.01, fixed before the 2048-box comparison. Each run has one initial seed per
independent box. The exact intervals follow [Clopper and Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404).
Two exact 97.5% binomial intervals give at least 95% joint
coverage by the union bound; subtracting their opposite endpoints gives a
conservative interval for the probability difference. This does not assume
independence between runs sharing random streams. The entire difference
interval must lie inside [-0.01, 0.01] to establish the declared tolerance.
An interval that crosses a tolerance boundary is inconclusive, even when
point estimates agree. Any larger confirmatory ensemble must be declared
before its outcomes are examined; pilot estimates alone are not a stopping
rule for nominal confidence coverage.

All six hosted workflows passed on `5aa5b47ff`: test, linting, documentation,
assistant plugins, PyPI packaging and Conda feedstock checks. These are
validation results, not a package release or merge.


The 2048-box pilot gave 39 reached at dt=1 s and 34 at dt=0.5 s, with no
unresolved lineages. The conservative difference interval is
[-0.016076, 0.011208], so the pilot does **not** resolve the declared tolerance.
A fixed confirmation is declared before execution: 8192 independent boxes
per timestep, new root seed 318, dt=1 and 0.5 s, all physical defaults and
20,000 s duration unchanged. Its intervals will be assessed once, separately
from the pilot; neither pooling nor repeated nominal-95% stopping is used.


| Neutral pilot | Reached / 2048 | Maximum mass residual | Synchronized stepping |
| --- | --- | --- | --- |
| dt=1 s, capacity 128 | 39 | 6.22e-15 | 205.7 s |
| dt=0.5 s, capacity 128 | 34 | 5.56e-15 | 407.8 s |
| dt=1 s, capacity 256 | 39 | 6.22e-15 | 304.0 s |

The capacity comparison preserves the complete cohort outcome time series
exactly; maximum observed occupancy was 33 in all runs. These single-run
stepping times exclude preparation, explicit gas forcing and analysis. They
are workload observations on RTX 5060, not a matched CPU speedup or a repeated
performance benchmark. Ion-production capacity and cleanup sensitivity are
still outstanding.


The updated interval helper passed all 21 study-helper tests on host CPU/CUDA
(`.venv/bin/pytest scripts/tests/lucky_few_study_test.py -q -Werror --no-cov`).
Tests cover exact binomial endpoints, sign reversal of difference intervals,
invalid confidence levels and controlled condensation with inventory closure.
Ruff and formatting passed; documentation structure had zero errors and the
website built 453 pages with no link errors. The full-package result above
still predates periodic cleanup; a final complete run remains required.


## Full-package follow-through after periodic cleanup

At library revision `d14f0899c` (with documentation and study-helper review
changes in the working tree), host CPU/CUDA validation completed:

| Command | Result |
| --- | --- |
| `WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/pytest --cov=particula --cov-report=term-missing --cov-fail-under=80 -Werror` | 10,119 passed, 14 skipped, 1 xfailed; 91.85% coverage |
| `.venv/bin/ruff check particula examples_tests scripts --no-fix` | Passed |
| `.venv/bin/ruff format particula examples_tests scripts --check` | 646 files formatted |
| `.venv/bin/mypy particula --ignore-missing-imports` | Passed, 586 files |

This full-package result includes the periodic-cleanup implementation and
supersedes the earlier package run that preceded it. The CUDA device was
available on the host; it was not replaced with a CPU fallback. The test run
overlapped the 8192-box half-step confirmation from 06:53 UTC on 2026-11-14.
That confirmation's stepping duration is therefore excluded from performance
comparisons. Numerical outcomes and conservation assertions remain separate
from timing claims. All six hosted workflows passed on `d14f0899c`; later
uncommitted example/helper changes still require their own follow-through.


## Example review and composed ion checks

The ion-balance reference now rejects nonuniform/nonzero-origin time grids
instead of silently evaluating different times, and rejects invalid chain
parameters. Ten independent tests include analytic two-pair decay: rates
4 alpha/V then alpha/V yield P2=exp(-4 alpha t/V) and
P1=(4/3)[exp(-alpha t/V)-P2], from which both moments are checked.

The complete example/tooling command, `WARP_CACHE_PATH=/tmp/particula-charged-warp
.venv/bin/pytest examples_tests scripts/tests -q -Werror --no-cov`, passed
493 tests with one skip on the host. This includes controlled growth under
both aerosol process orders on Warp CPU and actual CUDA. The full ion-balance
CUDA refresh is `run-20261114T070049797164Z`; its inspected figure is
byte-identical to the prior numerical figure. The executed source and built
Python download both match the maintained source and compile. The changed
example check passed, website tests passed (3), and the 453-page build reported
zero link errors. CUDA-policy skips in the routine example checker are distinct
from the deliberately refreshed CUDA workload.

The selected environment retains installed Particula distribution metadata
0.2.14 while the executed checkout reports source version 0.3.0. The manifest
records installed distribution versions separately from the actual source
revision and package-source hash; the environment was preserved.

Composed-system CUDA checks used 64 boxes, root seed 319, dt=1 s, cleanup every
10 s and all frozen physical defaults for 20,000 s:

| q [pairs cm^-3 s^-1] | Omission assumption | Capacity | Reached / captured / unresolved | Maximum mass residual | Observed peak slots |
| --- | --- | --- | --- | --- | --- |
| 2 | lower | 128 | 1 / 63 / 0 | 4.00e-15 | 34 |
| 2 | upper | 128 | 1 / 63 / 0 | 4.00e-15 | 34 |
| 10 | lower | 128 | 1 / 63 / 0 | 3.78e-15 | 35 |
| 10 | upper | 128 | 1 / 63 / 0 | 3.78e-15 | 35 |
| 10 | lower | 256 | 1 / 63 / 0 | 3.78e-15 | 35 |
| 10 | upper | 256 | 1 / 63 / 0 | 3.78e-15 | 35 |

Both q=10 capacity comparisons preserved the complete cohort time series
exactly. These are composed-path and capacity checks under each implemented
assumption, not production model selection. The ensemble is too small to
establish an ion effect or insensitivity of rare survival. Coefficient endpoints
do not necessarily bound survival outcomes in this nonlinear coupled model.
These runs and the example refresh overlapped the neutral confirmation; their
timings are not performance qualification.

At this stage the survival manuscript's plots had been inspected using the
actual neutral pilot. The manuscript has since moved to
`docs/examples/gpu/prepared/gpu_lucky_few.py` for the accepted two-assumption
sensitivity demonstration. Charged qualification and publication evidence
are recorded below as they become available.


## Fixed neutral timestep confirmation

The predeclared confirmation used 8192 independent boxes, root seed 318 and
20,000 s at the frozen physical defaults. Both cases used the helper loaded
at `d14f0899c`, before the order-control option was added. Their specification
recorded helper SHA256
`14f936ca87a115653c73b3dc8d36af14793473b264cd7a142ee50eb77918361e`.

| Timestep | Reached | Captured | Unresolved | Exact 95% success interval | Maximum mass residual |
| --- | --- | --- | --- | --- | --- |
| 1 s | 153 | 8039 | 0 | [0.0158563, 0.0218466] | 5.78e-15 |
| 0.5 s | 159 | 8033 | 0 | [0.0165326, 0.0226344] | 5.33e-15 |

The conservative 95% difference interval (half-step minus default) is
[-0.00616579, 0.00762977]. It lies inside the predeclared absolute tolerance
[-0.01, 0.01], so the neutral timestep check passes at that resolution. This
is not proof that numerical differences smaller than one percentage point
are negligible, nor does it qualify charged ion-production conditions. Pilot
and confirmation samples were not pooled, and the fresh-seed confirmation
was assessed once. The half-step timing overlapped other CUDA validation
and is excluded from performance comparisons.


## Neutral process-order confirmation and reference review

The separately predeclared order check used `ee8384eec`, 8192 boxes, root
seed 318, dt=1 s and 20,000 s. The script asserted that every configuration
field except aerosol process order matched the saved confirmation baseline.
With coagulation before condensation, it recorded 152 reached, 8040 captured
and zero unresolved lineages. Its exact 95% success interval was
[0.0157437, 0.0217152], and maximum mass residual was 6.67e-15.

The conservative 95% difference interval versus the default order is
[-0.00694530, 0.00670130], inside the fixed [-0.01, 0.01] tolerance. This
qualifies the neutral ordering comparison at that absolute resolution.
Timestep and ordering intervals are separate, not simultaneous 95% bounds
across every numerical comparison. Charged-production and cleanup-frequency
qualification remain outstanding. Overlapping runs exclude timing claims.

Source review found and corrected one remaining incorrect D'yachkov citation
in `charged_kernel_wp`. The [publisher record](https://doi.org/10.1063/1.2713719)
identifies D'yachkov, the two Khrapaks and Morfill in *Physics of Plasmas* 14,
042102 (2007). The common pair scaling and hard-sphere fit are separately
attributed to [Gopalakrishnan and Hogan (2012)](https://doi.org/10.1103/PhysRevE.85.026410),
with Gatti/Kortshagen and Chahl/Gopalakrishnan references included for the other
selected fits. After removing docstrings, the changed GPU module has an
identical executable AST to `ee8384eec`.

The manuscript now states the implemented interaction boundary explicitly:
aerosol–aerosol electrostatics enter through the product of the charges, with
no additional image/polarization term. At fixed masses and radii, either zero
aerosol charge removes that Coulomb correction. LF2013 ion attachment remains
a separate collision law. These model boundaries limit interpretation of any
future atmospheric-survival comparison.

## Accepted two-assumption qualification campaign

The accepted study compares the documented omissions replaced by 0 and 1e-15
m^3/s. Settings were frozen before execution: 8192 boxes, root seed 320,
20,000 s horizon, dt=1 s, capacity 128, cleanup every 10 s, q=2/10 pairs
cm^-3 s^-1. For each assumption and q, the baseline is compared with dt=0.5 s,
reversed aerosol order, cleanup every 5 s and capacity 256. Each conservative
95% probability-difference interval must lie within [-0.01, 0.01]; each
baseline exact 95% interval must have width <=0.01. These cover individual
comparisons, not a simultaneous confidence guarantee across the matrix.

Run the fixed matrix with:

```bash
WARP_CACHE_PATH=/tmp/particula-charged-warp .venv/bin/python \
  -m scripts.gpu_studies.qualify_lucky_few --output /tmp/lucky-few-qualification
```

The output directory must be new. The specification records the source revision
and helper hash before any run; reduced results and checks are saved after each
case. The script retains failed or inconclusive comparisons and exits nonzero
if any acceptance criterion is unresolved.

`stop_when_resolved=True` ends a case only when every original lineage has an
absorbing reached/captured outcome. It preserves survival at the configured
horizon but does not predict later aerosol evolution. Stopped runs matched the
full-run prefix and final outcomes on CPU/CUDA; a censored case still ran to its
horizon. The focused helper suite passed 25 tests. The complete example/tooling
suite passed 494 tests with one skip using host CUDA and the writable Warp cache.
A preceding sandbox run failed only on its read-only default Warp cache.

The registered source is linked from the GPU index, prepared guide and website
navigation. A preliminary build produced 454 pages and 584 HTML documents with
zero link errors. The extracted helper ZIP imported its own local helper and
executed all five small CUDA cases outside the checkout with `PYTHONPATH`
removed. Its first smoke assertion used exact decimal-time equality; rerunning
with the established 1e-14 time tolerance passed. This download smoke check is
separate from the complete publication workload and its numerical qualification.

## Completed charged survival qualification

All 20 CUDA runs completed and all predeclared checks passed. The four baseline
precision checks and 16 numerical comparisons use individual confidence
intervals; they do not form a simultaneous 95% guarantee over the matrix.
The accepted resolution is one percentage point in absolute survival
probability. Smaller ion effects are not qualified by this criterion.

The campaign used library revision `58ab7e3e870b08932149baca4ae5c62a5ba99664` and the
study helper now committed in `f861f9fd4`, SHA256
`5f61eb0415b560e577b8f6d841053c3cf8eda7e9a4a3acb184812663d9ec8ca1`.
All cases resolved every lineage before the 20,000 s horizon. Their last
diagnostics ranged from 3710 to 3790 s. Maximum per-species mass residual was
2.998e-15, exact net charge remained zero, and peak observed occupancy
was 37 slots. Diagnostic occupancy is not an instantaneous upper bound.

| q [pairs cm^-3 s^-1] | Omitted coefficient [m^3/s] | Baseline reached / 8192 | Exact 95% success interval |
| --- | --- | --- | --- |
| 2 | 0 | 140 | [0.014395, 0.020136] |
| 2 | 1e-15 | 140 | [0.014395, 0.020136] |
| 10 | 0 | 151 | [0.015631, 0.021584] |
| 10 | 1e-15 | 151 | [0.015631, 0.021584] |

Differences below are variant minus its matching baseline.

| q | Omitted coefficient [m^3/s] | Numerical change | Reached / 8192 | Conservative 95% difference interval |
| --- | --- | --- | --- | --- |
| 2 | 0 | half-step | 135 | [-0.007102, 0.005882] |
| 2 | 0 | reversed-order | 140 | [-0.006549, 0.006549] |
| 2 | 0 | half-cleanup | 140 | [-0.006549, 0.006549] |
| 2 | 0 | double-capacity | 140 | [-0.006549, 0.006549] |
| 2 | 1e-15 | half-step | 135 | [-0.007102, 0.005882] |
| 2 | 1e-15 | reversed-order | 140 | [-0.006549, 0.006549] |
| 2 | 1e-15 | half-cleanup | 140 | [-0.006549, 0.006549] |
| 2 | 1e-15 | double-capacity | 140 | [-0.006549, 0.006549] |
| 10 | 0 | half-step | 153 | [-0.006569, 0.007057] |
| 10 | 0 | reversed-order | 152 | [-0.006680, 0.006924] |
| 10 | 0 | half-cleanup | 150 | [-0.006902, 0.006658] |
| 10 | 0 | double-capacity | 151 | [-0.006791, 0.006791] |
| 10 | 1e-15 | half-step | 153 | [-0.006569, 0.007057] |
| 10 | 1e-15 | reversed-order | 152 | [-0.006680, 0.006924] |
| 10 | 1e-15 | half-cleanup | 150 | [-0.006902, 0.006658] |
| 10 | 1e-15 | double-capacity | 151 | [-0.006791, 0.006791] |

All four doubled-capacity runs preserved the complete recorded outcome time
series exactly. Nine earlier full-horizon validation histories also retained
identical outcome counts after every lineage had resolved, supporting the
absorbing-outcome stopping rule alongside the CPU/CUDA regressions.

Short CUDA regression and download-smoke jobs overlapped early campaign runs;
these timings are not performance comparisons. The publication workload uses
a separate fixed root seed, 321, with the qualified default settings. Both
coefficient assumptions remain explicit sensitivity cases, not bounds on
survival probability or an empirical atmospheric validation.

## Complete Lucky Few publication run

The complete five-case CUDA refresh is
`run-20261114T140715836273Z`, executed from clean source revision
`4a550421e7c0f01f4f0e6d2e9a9a75e8648c4b9f` on the RTX 5060.
It used 8192 boxes per case, root seed 321, and the qualified defaults.
The saved source matches the maintained percent script byte for byte; both
helper hashes match the manifest. The 300-DPI three-panel figure was visually
reviewed with the actual five publication cases.

| q [pairs cm^-3 s^-1] | Omitted coefficient [m^3/s] | Reached | Captured | Unresolved | Exact 95% success interval |
| --- | --- | --- | --- | --- | --- |
| 0 | strict; no ions | 163 | 8029 | 0 | [0.016984, 0.023159] |
| 2 | 0 | 154 | 8038 | 0 | [0.015969, 0.021978] |
| 2 | 1e-15 | 154 | 8038 | 0 | [0.015969, 0.021978] |
| 10 | 0 | 163 | 8029 | 0 | [0.016984, 0.023159] |
| 10 | 1e-15 | 163 | 8029 | 0 | [0.016984, 0.023159] |

The conservative 95% ion-minus-control intervals include zero for every
comparison. This run does not resolve an ion-production effect. Matching
coefficient-case outcomes do not establish identical survival probabilities
or mathematical bounds. These are illustrative finite-box results, not
field-validated atmospheric predictions.

All reported maximum mass residuals were 2.89e-15, exact charge checks passed,
and observed peak occupancy was at most 37/128 slots. Final diagnostics were
at 3710–3740 s, with no unresolved lineages. The complete refresh took
975.98 s (about 16.3 minutes). Printed stepping times exclude
diagnostic transfers, plotting and analysis; preparation timing begins after
CPU initial-state construction. These are execution diagnostics, not a
matched CPU/GPU speedup measurement.

Final download checks compared the built Python and ZIP sources with the
maintained source and both declared helpers byte for byte. The extracted ZIP
passed all five small CUDA smoke cases outside the checkout with `PYTHONPATH`
removed. The changed-example check passed; its CUDA policy skips are separate
from the deliberate full CUDA refresh above. Website tests passed (3), and
the final build produced 454 pages / 584 HTML documents / 270 API modules with
zero link errors. Final hosted status is recorded on
[PR #1650](https://github.com/uncscode/particula/pull/1650/checks), separately
from local CUDA evidence.
