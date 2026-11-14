# Charged GPU coagulation and the Lucky Few

Accepted plan, 2026-11-13. Branch:
`codex/gpu-charged-coagulation-lucky-few`; base `upstream/main`.
Delivery: validated implementation, saved CUDA examples, fork push and draft
upstream PR. Merge and release are outside this authorization.

## Completion ledger

- [x] Verify clean checkout/current upstream and create dedicated branch.
- [x] Port hard-sphere, Dyachkov 2007, Gatti 2008, Gopalakrishnan 2012,
  and Chahl 2019 charged kernels using fp64 and the shared sampler.
- [x] Integrate selection into prepared execution, parallel majorants,
  capture and exact checkpoint continuation without double-counting Brownian.
- [x] Add explicit ion pair classification, LF2013 attachment and atmospheric
  recombination with validated domains and no implicit extrapolation.
- [x] Add paired Poisson ion production, source inventory ledgers, fixed
  capacity and independent persistent source RNG with capture/restart.
- [x] Add neutral-product cleanup and exported inventory ledgers as a prepared
  process; the study injects ions first and performs cleanup last.
- [x] Exclude free ions and neutral products from condensation proposals;
  preserve uptake by aerosol particles with attached ions and exact inventories.
- [x] Add resident cohort metadata, first-passage tracking and censoring;
  study-level timestep and aerosol process-order qualification is recorded below.
- [x] Validate implemented library formulas, stochastic collisions and lifecycle
  behavior on native CPU references, Warp CPU and actual CUDA.
- [x] Complete survival-study convergence and uncertainty qualification.
- [x] Add narrated percent-cell charged model comparison and saved CUDA output.
- [x] Add atmospheric ion-balance percent source and inspect saved CUDA output.
- [x] Add the 3–50 nm Lucky Few percent-cell study.
- [x] Execute full examples, inspect saved figures, verify downloads/build.
- [x] Open draft upstream PR #1650 and push validated implementation increments.
- [x] Finish the survival example and final diff review.

Final hosted CI is tracked on the [current PR head](https://github.com/uncscode/particula/pull/1650/checks).
All required workflows must pass before task closeout; these external results
are separate from the saved local CUDA evidence.

## Scientific contract

Explicit ions have diameter 0.5 nm, unit represented counts and charges ±1 e.
Use 298.15 K, 101325 Pa, effective positive/negative masses 150/90 u and
consistent densities. Constant production cases are 0, 2 and 10 ion pairs
cm^-3 s^-1. Each Poisson event adds both polarities; mean is q V dt.
The bounded recombination reference is 1.7e-12 m^3/s. Atmospheric validation
means agreement with cited attachment/recombination references, not field
validation of a complete atmospheric chemistry model.

Ion–aerosol collisions use LF2013 within its actual material, temperature,
pressure, charge and radius bounds. Opposite free ions form neutral products
that cannot collide or condense and are removed at refresh with exported mass
accounted. Same-polarity free ions do not form stable clusters in this model.
Free ions do not undergo ordinary vapor condensation; attached ion mass remains.
Do not remove neutralized aerosol particles. Never apply bulk recombination or
attachment sinks on top of explicit collisions.

Prepared charged models are alternatives to Brownian, not additive corrections.
Preserve existing explicitly additive direct API behavior and default selection.
No full pair matrix, hidden transfers/fallback, unequal weights, resizing,
resampling, or volume changes in the study.

Primary outcome: fraction of original 3 nm seed lineages reaching 50 nm before
any background merger. Seed–seed merges retain unresolved lineage counts;
background capture ends eligibility. Count first passage only once, report
particle counts separately, and retain censored lineages at the end.

## Evidence and examples

Controlled single-vapor growth, not full atmospheric chemistry. Use independent
boxes for uncertainty. Freeze timestep, cleanup interval, capacity and ensemble
size after a bounded convergence pilot; do not tune for a desired ion effect.
Report upper bounds for zero events and synchronized GPU timings separately
from preparation/transfers/analysis. CPU speedup requires matched workloads.

Test independent formula values and CPU/Warp/CUDA parity; physical limits;
majorants and collision statistics; source/cleanup mass and charge budgets;
ion equilibrium and decay; lineage merges/capture/censoring; capacity rejection;
resource identities, RNG, faults, capture and checkpoint continuation.
Keep existing conservation tolerances and predeclare acceptance thresholds.

Register three # %% / # %% [markdown] sources under docs/examples/gpu/ in
website/examples.json. Refresh complete CUDA workloads and retain compact
website/results bundles, not particle state dumps. Check extracted downloads,
documentation structure, website tests/build, Ruff, mypy, package/example/tooling
suites and the repository coverage gate. CUDA skips are not qualification.
Update owning GPU guides and validation evidence; preserve README boundaries.

## Primary references

- Ion production: https://doi.org/10.1029/2009JD012630
- Ion balance: https://doi.org/10.5194/acp-13-11573-2013
- Recombination: https://doi.org/10.5194/acp-22-12443-2022
- LF2013 attachment: https://doi.org/10.1080/02786826.2013.783684

## Implementation evidence so far

- GPU verified on host: RTX 5060, 8151 MiB; restricted sandbox hides driver.
- Charged/direct/prepared regression: 942 passed on host CPU and CUDA.
- New charged formulas, dimensional majorants, capture/restart and strict
  attachment checks: 43 passed on host CPU and CUDA.
- First narrated CUDA model-comparison example executed and visually inspected.
- Initial source increment added the paired source and ledger; subsequent
  increments below add cleanup, lineages and atmospheric collision integration.
- Source, capture/restart and charged lifecycle checks: 62 passed on host
  CPU/CUDA; subsequent composition and zero-production tests also passed.
- Example and tooling suites: 459 passed, 1 skipped with host cache access.
- Earlier full package run: 10,047 passed, 14 skipped, 1 xfailed; 92.04% coverage
  with actual CUDA available. Ruff, formatting, mypy and website build pass.
- Both existing charged examples refreshed on the host and figures inspected.
- Full documentation example check passed; saved bundles are current, CPU-policy
  examples executed, and the new CUDA comparison was explicitly refreshed.
- Neutral ion-product cleanup, export accounting, capture/restart and source
  regression: 58 passed on host CPU/CUDA. Reference review corrected existing
  Dyachkov bibliographic errors and added primary papers to the CUDA example.
- Seed-lineage observer and affected prepared regression: 171 passed on host
  CPU/CUDA, including actual condensation growth, merger/capture precedence,
  native capture and exact continuation before/after mergers.
- LF2013 device helpers use algebraically centered published polynomials;
  all 201 charge states and both polarities match 70-digit decimal evaluation
  at nine radii per valid interval within 1e-10 relative tolerance. These helpers
  are now integrated into the atmospheric collision scheduler.
- Ion exclusion and mixed-density Kelvin support: 171 combined ion-physics and
  prepared tests passed on host CPU/CUDA, including native CPU parity at 2e-12.
- Discovered scientific gap: LF2013 omits repulsive attachment rates below
  1e-15 m^3/s. For same-sign singly charged seeds, the fit begins at radius
  5.30835917 nm, above the planned 1.5 nm seed radius. Production policy is
  awaiting the user's choice between explicit uncertainty bracketing and a
  deeper attachment-theory implementation; default coefficient lookup rejects.

- Atmospheric pair routing and lifecycle tests: 30 tests included in the full
  package run, with CPU/CUDA recombination, attachment, omission bounds,
  fault-before-RNG behavior, source/cleanup/cohort composition and restart.
- Prepared/parallel/formula regression: 1,721 passed on host CPU/CUDA.
- Full package after atmospheric integration: 10,110 passed, 14 skipped,
  1 xfailed; 91.85% coverage with actual CUDA and warnings treated as errors.
- Ion-balance CUDA example: 64 boxes per production condition, 2000 simulated
  seconds, exact finite-population birth/death reference; maximum mean errors
  1.58/2.67/2.54 standard errors for q=0/2/10. Mass residuals below 1e-14.
  Plot reviewed; source download matches and compiles; website build passed.

- Added configurable neutral-product cleanup intervals with checkpointed phase;
  104 affected CPU/CUDA lifecycle tests passed. The survival pilot can now vary
  cleanup frequency independently of its physical timestep.
- Fixed the hosted documentation reachability failure by linking the ion-balance
  source from its owning index and guide; committed-file structure and 13
  structure tests pass. Hosted follow-through remains required on the final head.

## Survival pilot specification and first neutral result

The reusable runner is `scripts/gpu_studies/lucky_few.py`. Physical defaults
were set before the first neutral pilot and are frozen for subsequent ion
comparisons: one 3 nm original seed and 32 neutral 100 nm background particles
per 3.2e-10 m³ box, material density 1400 kg/m³, target diameter 50 nm, fixed
298.15 K and 101325 Pa. The illustrative vapor has molar mass 0.2 kg/mol,
diffusivity 5e-6 m²/s, reference vapor pressure 1e-12 Pa and unit accommodation.
Its concentration is replenished to 2e-10 kg/m³ every 10 s. This is a defined
pulsed feed, not an assertion of constant vapor concentration between updates.
Every signed gas-inventory change is included in the external feed budget.

Vapor growth uses continuous deterministic condensation. Collision selection,
ion production and charge histories are stochastic. This does not implement
molecule-by-molecule stochastic vapor growth or ion-induced nucleation.
Background particles coagulate and grow; they are not an immutable sink.
One initial seed per independent box supports exact binomial intervals for
reaching the threshold by the finite observation time. Censored lineages remain
explicit. A zero-event result additionally reports a one-sided 95% upper bound.

First CUDA pilot: q=0, 64 boxes, capacity 128, dt=1 s, cleanup every 10 s,
duration 20,000 s, root seed 317. Outcomes were 2 reached, 62 captured and
0 unresolved. Exact two-sided 95% success interval: 0.003807–0.108371.
Maximum relative mass residual: 4.67e-15; maximum observed occupied slots: 33.
Synchronized stepping took 36.1 s on RTX 5060, separate from 9.34 s preparation
and 0.382 s gas forcing; these are one-run observations, not a speedup benchmark.
The 125-test combined helper/ion/lifecycle suite passed on CPU and CUDA.

The no-background growth control reached 50 nm for all four seeds at dt =
2, 1 and 0.5 s, first observed at 1390 s in each run (10 s observation spacing).
Maximum mass residual was below 3.45e-15. The predeclared timing criterion passed.

The 2048-box neutral timestep/capacity pilot is complete. Its predeclared
absolute probability tolerance is 0.01, assessed using a conservative 95%
difference interval, not only point estimates. A larger confirmatory ensemble
may be needed if this pilot cannot resolve that tolerance.
Then the production ion cases require the outstanding omitted-rate choice.
Keep the physical defaults fixed; do not tune a desired charging benefit.

Additional primary context for the final percent-cell narrative:
- Lehtinen et al. (2007), growth versus scavenging and apparent formation rates:
  https://doi.org/10.1016/j.jaerosci.2007.06.009.
- Atmospheric Nanoparticle Survivability Reduction Due to Charge-Induced
  Coagulation Scavenging Enhancement (2021):
  https://doi.org/10.1029/2021GL092758. Charging need not improve survival.
- Robust metric for quantifying the importance of stochastic effects on
  nanoparticle growth (2018): https://doi.org/10.1038/s41598-018-32610-z.
  Distinguish discrete molecular growth from the continuous condensation used here.


The 2048-box pilot gave 39 reached at dt=1 s and 34 at dt=0.5 s, with no
unresolved lineages. The conservative difference interval is
[-0.016076, 0.011208], so the pilot does **not** resolve the declared tolerance.
A fixed confirmation is declared before execution: 8192 independent boxes
per timestep, new root seed 318, dt=1 and 0.5 s, all physical defaults and
20,000 s duration unchanged. Its intervals will be assessed once, separately
from the pilot; neither pooling nor repeated nominal-95% stopping is used.


Doubling capacity from 128 to 256 preserved the full neutral cohort time
series exactly (39 reached, 2009 captured). All three pilot runs had mass
residuals below 6.3e-15 and maximum observed occupancy 33. This establishes
neutral capacity invariance for this workload, not the ion-production cases.


The third percent-cell manuscript is staged at `plans/gpu_lucky_few_example.py`
until its numerical settings and omitted-rate policy are qualified. It has
explicit policy/numerical CLI arguments, unchanged physical defaults, cohort
curves, final exact intervals, outcome partitions, budget/timing output and
primary references. Its plotting layout was inspected using the actual neutral
2048-box pilot, not fabricated ion results. It is not yet a published or fully
executed example and does not satisfy the unchecked final-example gates.

The manuscript distinguishes lower/upper **attachment coefficient** assumptions
from survival-probability bounds: nonlinear charge and collision histories mean
endpoint coefficient runs do not guarantee extrema of the survival outcome.


The study helper now exposes `coagulation_first` for the required process-order
check, retaining the existing order by default and keeping source first and
cleanup last. Both orders pass the controlled-growth check on Warp CPU and actual CUDA.
Twenty pure helper checks also pass. The existing 8192-box confirmation retains its original
loaded helper, revision `d14f0899c` and recorded source hash.

Review of the ion-balance percent source found that its reference solver
silently assumed a uniform time grid. The source now rejects incompatible
sampling and invalid chain parameters. Ten independent example-reference tests
pass, including the analytic two-pair pure-death mean and variance. Its complete
CUDA refresh is now saved as `run-20261114T070049797164Z`; the numerical figure
is byte-identical to the prior run and was visually inspected.


Neutral order qualification is predeclared separately from the timestep test:
8192 boxes, root seed 318, dt=1 s, coagulation before condensation, compared
with the confirmation baseline. All other physical and numerical parameters
must match exactly; a 95% difference interval must lie inside [-0.01, 0.01].
This is a separate interval, not a claim of simultaneous 95% coverage across
all numerical comparisons. Execution is queued conceptually after the active
GPU workload; no order-study process has been launched yet.


Independent composed-system validation will exercise both implemented omission
configurations at q=2 and 10 with 64 boxes, root seed 319 and the frozen physical
defaults, then compare q=10 capacity 128 versus 256 for each assumption. These
are integration checks of supported paths, not production policy selection or
qualification of rare survival differences. Full-package correctness testing
began alongside the half-step confirmation at 2026-11-14 06:53 UTC; that
confirmation's elapsed stepping time will not be used as performance evidence.


Full package after periodic cleanup: 10,119 passed, 14 skipped, 1 xfailed;
91.85% coverage on host CPU/CUDA with warnings treated as errors. Ruff,
formatting (646 files) and mypy (586 files) pass. All six hosted workflows
passed on `d14f0899c`. The reference/manuscript/order-control increment passed the complete
example/tooling suite: 493 passed, 1 skipped, with actual CUDA. The changed
example check passed; CUDA examples retain explicit-refresh policy. The website
built 453 pages with no link errors, and the ion-balance download matches and
compiles. Final hosted follow-through remains required after this increment.


Before moving the manuscript into published examples, freeze the qualified
numerical/policy defaults so the downloadable bundle's `python example.py
--cuda` invocation runs the complete study. Register its helper as an input
and test the extracted ZIP independently. The draft deliberately still
requires explicit settings and is not a replacement for those publication gates.


The six composed ion integration runs completed: q=2/10 under each explicit
omission assumption, plus q=10 at doubled capacity. Each recorded 1 reached,
63 captured and no unresolved lineages, with maximum mass residual below
4e-15. Observed occupied slots peaked at 34 for q=2 and 35 for q=10. Both
capacity comparisons preserved the full cohort time series exactly. These
small runs validate composed code paths; they neither establish a rare-event
ion effect nor choose the production omitted-rate policy.


The fixed 8192-box neutral timestep confirmation completed at the predeclared
root seed 318. The default step reached 153 seeds (8039 captured); the half
step reached 159 (8033 captured); neither left unresolved lineages. The 95%
difference interval, half-step minus default, is [-0.006166, 0.007630], wholly
inside the fixed [-0.01, 0.01] tolerance. This qualifies the neutral timestep
comparison to that absolute resolution, not arbitrary smaller effects or
ion-production cases. Mass residuals were below 5.8e-15. The separate order
run is active on actual CUDA; the production omission-policy choice remains
pending.


The 8192-box neutral process-order check also completed. Coagulation first
produced 152 reached, 8040 captured and no unresolved lineages; its 95%
difference interval versus the default order is [-0.006945, 0.006701], inside
the predeclared [-0.01, 0.01] tolerance. Maximum mass residual was 6.67e-15.
The neutral timestep and order requirements are qualified at one-percentage-
point absolute resolution. Charged production timestep/order/cleanup and
ensemble qualification remain open; the 64-box ion checks do not replace them.

Source review corrected the remaining erroneous D'yachkov bibliography entry
in the GPU pair primitive, added the other model references, and documented
its Coulomb-only aerosol interaction. The manuscript now explicitly excludes
an additional aerosol image/polarization force and cites the hard-sphere fit.
The executable GPU AST is unchanged after removing docstrings; these edits
change documentation, not the numerical implementation.

## Accepted attachment sensitivity and final qualification

On 2026-11-14 the user selected both explicit omitted-rate assumptions, zero
and 1e-15 m^3/s. This resolves the production-policy decision; the two cases
remain sensitivity assumptions, not bounds on survival probability.

The final charged matrix is frozen before execution: q=2/10, lower/upper,
8192 boxes, root seed 320, 20,000 s horizon, dt=1 s, cleanup=10 s and capacity
128. Each case is compared with dt=0.5 s, reversed aerosol process order,
cleanup=5 s and capacity 256. Every separate conservative 95% probability
difference interval must lie inside [-0.01, 0.01]; each baseline exact 95%
interval must have width <=0.01. These are individual comparison coverages,
not a joint confidence statement for the complete matrix. No tolerance will
be adjusted after observing results. The reusable runner is
`scripts/gpu_studies/qualify_lucky_few.py`.

An explicit helper option stops at the first diagnostic with no unresolved
lineages. Reached/captured are absorbing outcomes, so this preserves the
specified horizon's survival outcome without simulating later aerosol state.
CPU/CUDA regressions compare the stopped prefix/final outcome with full runs
and retain unresolved lineages through the horizon: 25 helper tests passed.
The earlier full-horizon runs resolved all lineages by 3730 s; this observation
motivates the optimization, not a shorter fixed observation horizon.

## Charged qualification completed

All 20 fixed CUDA runs passed: four baseline sampling-precision checks and
16 timestep/order/cleanup/capacity comparisons. Every individual conservative
95% difference interval lies within the predeclared [-0.01, 0.01] tolerance.
Baseline successes were 140/8192 for q=2 and 151/8192 for q=10 under either
coefficient assumption. All lineages resolved, maximum mass residual was
2.998e-15, and all four capacity time series matched their baselines exactly.
The complete matrix, source hash and limitations are in the validation record.
The separate root-321 publication run and final saved-output/CI checks remain.

## Published example and local closeout

The complete CUDA generation is `run-20261114T140715836273Z`, from source
commit `4a550421e7c0f01f4f0e6d2e9a9a75e8648c4b9f`. With root seed 321 and
8192 boxes per case, the no-ion control reached 163 lineages; q=2 reached 154
and q=10 reached 163 under either omission assumption. Every lineage resolved.
The conservative ion-minus-control intervals include zero; this is a GPU
sensitivity demonstration, not a resolved ion effect or atmospheric validation.

The actual saved figure was inspected. Maintained source, saved executed source,
built Python download and extracted ZIP source match; declared helper hashes
match, and the extracted ZIP passed all five small CUDA smoke cases outside the
checkout. The changed-example check passed, all three website tests passed,
and the full site build produced 454 pages / 584 HTML files with no link errors.
The final host example/tooling suite passed 494 tests with one skip; the new
stopping helper has 25 passing CPU/CUDA tests. Source and references received
a final review, with no additional actionable findings.

Upstream advanced from the original base to `e5f0af936` only by changing the
website Mistune lock from 3.1.4 to 3.3.3. Simulator code and the selected local
research environment are unchanged. Final PR CI checks integration with the
current upstream base. Merge and release remain outside this task.

## Additional two-million-particle environmental example

The user selected charged aerosols with temperature, pressure and humidity
variation, separate from the fixed-reference explicit-ion study. Design:
16 controlled environments (278.15/298.15 K, 70000/101325 Pa, initial RH
0.4/0.6/0.8/0.9), 1024 independent boxes per condition and 128 initially
active particles per box: 2,097,152 simultaneous resolved particles.
Unit counts, matched polydisperse initial populations, balanced charge,
closed water vapor plus kappa-water condensation and charged Brownian
coagulation. Vapor diffusivity is an explicit fixed 2e-5 m2/s approximation.
There is no altitude trajectory, ion attachment or humidity replenishment.

- [x] Existing charged feature passed all six hosted workflows at 4fefc50c3.
- [x] Small CUDA pilot: finite nonnegative state and per-box mass/charge budgets.
- [x] Choose 0.001 s / 1000 steps after full-layout performance and timestep pilots.
- [x] Validate water-growth timestep sensitivity and independent equilibrium law (nine tests passed on CPU/CUDA).
- [x] Run the complete 2,097,152-particle workload and inspect saved figures.
- [x] Register narrated percent source, test downloadable helpers and build site.
- Final push and hosted validation use the live [PR CI gate](https://github.com/uncscode/particula/pull/1650/checks).

Two complete runs retained identical printed scientific results: 124 mergers,
2,097,028 final particles, exact charge conservation and a maximum per-box
species-mass residual of 1.688e-14. Each complete refresh took about 14 minutes.
The second fixes crowded chart labels without changing physics or seed.
Nine strengthened CPU/CUDA tests passed, and the extracted ZIP passed a
separate CUDA smoke check. Details and provenance are in the
[environmental validation record](../docs/contribute/validation/gpu/environment-sweep.md).

## Extended environmental dynamics and variable ion transport

User-approved extension: retain only final saved output per example; add size,
charge, loss and collision histories; dry 0–86 km and 2/5/10 atm chambers;
longer humidity study with active coagulation and explicit ions.

- [x] Dry 1000 s study: 19 conditions, 2,490,368 initial particles; independent
  64-box timestep comparison at 0.5/0.25 s passed the predeclared individual
  95% interval criterion of +/-2 percentage points of initial number.
- [x] Add opt-in `mobility` ion attachment, preserving LF2013 defaults. Use
  constant collision-cross-section low-field mobility scaling
  Z=Z_ref sqrt(T/298.15) 101325/P, ion friction e/Z, and the existing
  Coulomb-adjusted hard-sphere transition fit. Ion radius is contact geometry,
  not a proxy for ion mobility. Defaults are illustrative reference mobilities.
  This is a transport approximation, not LF2013 extrapolation or a new measured
  atmospheric fit. No image forces, hydration or evolving ion chemistry.
  Opposite-ion recombination remains an explicit user-supplied coefficient.
- [x] Validate independent continuum/ballistic limits, polarity, mobility law,
  CPU/Warp/CUDA parity, invalid-input rejection and prepared continuation.
- [x] Humidity: include free ions, source and neutral cleanup; distinguish
  aerosol mergers, attachment and recombination using population/material
  ledgers. Preserve every-box mass and total charge accounting.
- [x] Execute the full dry CUDA study, inspect all six figures, interpret
  measured trends and verify the independently extracted CUDA download.
  449,485 mergers; 18.04894% number loss; maximum mass residual 1.554e-15.
- [x] Qualify humidity timestep with the frozen 256-box, seed-413 confirmation:
  all original criteria passed; maximum ion-event difference 1.90430% against
  2%. Retain the failed 16-box pilot alongside the confirmation.
- [ ] Execute the final humidity CUDA study and corrected
  Lucky Few qualification/publication; inspect and interpret the final results.
- [ ] Run relevant regression suites, documentation build, push and check PR CI.

Physical-limit validation uncovered an inherited inversion of Gc/Gk in the
charged diffusive Knudsen number. Correct both CPU and Warp before final runs;
requalify dry stepping and refresh affected charged examples. The corrected
rates also exposed a CPU radius-bin sampler chain: orient sampled donors by
actual radius and exclude self-pairs before donor uniqueness. Keep the existing
no-spurious-calcite-merger regression. Replace the interpolation diagnostic's
arbitrary suppression threshold with independently derived charged flux limits;
retain its interpolation agreement tolerance. Existing tiny-Knudsen tests now
use zero absolute tolerance and analytic expectations.

Current source implementation: 6bdf3ebce. Complete host CPU/CUDA package suite:
10093 passed, 13 skipped, 57 deselected, 1 xfailed; coverage 91.60%.
Dry refinement passed again after normalization correction. The 16-box
humidity pilot failed its 2% ion population criterion (water/vapor passed);
retain the failure. The separate frozen 256-box, seed-413 confirmation passed
with unchanged tolerances before publication. No adaptive tolerance relaxation.
