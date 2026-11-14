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
  process; final ion-collision ordering remains part of study integration.
- [x] Exclude free ions and neutral products from condensation proposals;
  preserve uptake by aerosol particles with attached ions and exact inventories.
- [x] Add resident cohort metadata, first-passage tracking and censoring;
  study-level timestep and process-order convergence remains outstanding.
- [x] Validate implemented library formulas, stochastic collisions and lifecycle
  behavior on native CPU references, Warp CPU and actual CUDA.
- [ ] Complete survival-study convergence and uncertainty qualification.
- [x] Add narrated percent-cell charged model comparison and saved CUDA output.
- [x] Add atmospheric ion-balance percent source and inspect saved CUDA output.
- [ ] Add the 3–50 nm Lucky Few percent-cell study.
- [ ] Execute full examples, inspect saved figures, verify downloads/build.
- [x] Open draft upstream PR #1650 and push validated implementation increments.
- [ ] Finish the survival example, final diff review and final hosted CI.

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
