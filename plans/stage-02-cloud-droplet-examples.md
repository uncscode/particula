# Cloud and droplet integration follow-through

Status: implementation, full runs, refinement and output review complete locally.
Added to the active Stage 2A/2B objective by the user after core implementation
and validation. Later commits await direct publication authorization for the
existing Stage 2B branch and dependent draft
[PR #1660](https://github.com/uncscode/particula/pull/1660).

## Requested outcome

Update cloud and droplet simulations to use the new integration for stability,
and add examples showing where the choice matters. Preserve scientific inputs,
particle workloads, stochastic processes, species budgets and backend claims.
A successful run or conserved total mass alone does not prove improved accuracy.

## Inventory and implementation targets

- CPU single-cycle chamber: 1,000 seeds, wet/dry reset, condensation and
  stochastic wall removal; retained output explicitly reports dry-phase
  fixed-step oscillations. Replace its condensation stepping with qualified
  adaptive integration and correct event/output timing; preserve the wall-loss
  schedule and species accounting.
- CPU multi-cycle chamber: three 1,000-seed compositions and four full wet/dry
  cycles. Apply the same qualified integration policy throughout all branches.
- Respiratory droplet partitioning: twelve material lanes and a 20 s
  condensation/evaporation history. Preserve the example's parameter-provenance
  caveats and compare time refinement without claiming measured lifetimes.
- Biomass-burning cloud interaction: initial condensation and the subsequent
  3,600 s collision-model comparisons. Adapt deterministic condensation while
  keeping explicit stochastic collision schedules and their uncertainty limits.
- GPU cyclic chamber and related prepared forcing studies: inspect how adaptive
  deterministic condensation can be composed with the existing stochastic/source
  processes without changing their physical event schedule or concealing host
  transfers. Do not replace a required CUDA workload with CPU or remove physics.
- Add compact executable stability/accuracy demonstrations for wet-to-dry
  relaxation and finite-reservoir growth/depletion, with fixed-step comparisons,
  refined/independent references, separate budgets and work counters.

## Acceptance

- New integration is exercised by updated full cloud/droplet workloads, not only
  by a detached toy test. Existing particle counts and durations are retained.
- Reference uncertainty is below the stated observable error target; convergence
  is measured on radius/water inventory and not inferred from bounded values.
- Discontinuities land at explicit physical times; source/reset/loss budgets
  remain separate from closed condensation budgets.
- Adaptive rejection never consumes stochastic draws or emits intermediate
  output. Collision and wall-loss scheduling remains explicit.
- Full published examples are refreshed through `docs_examples.py refresh`,
  scientific assertions retained, resulting figures inspected and narrative
  numbers updated from actual results. Compact saved data obeys existing policy.
- New demonstrations have navigation/topic entries and reproducible downloads.
- Relevant native/CUDA tests, example/tooling checks and documentation build pass;
  all required example runs finish before the draft PR is marked ready.

## Current evidence

Core Stage 2A is PR #1659. Core Stage 2B is committed at `6ed378cb26` and published
in draft PR #1660. Its 158 focused workflow tests and 10679-test package run do
not yet qualify the example migrations above. The full-population RK23 probe reached 30 s wet in 1,725 accepted steps, then
needed 2,754 accepted and 2,605 rejected trials for the first 0.1 s after the dry
reset (51.16 s cumulative wall time). It was deliberately stopped after this
cost finding; the remaining dry interval is unmeasured, not a completed run.
A one-particle ROS2 diagnostic completed both phases; this is method screening,
not a replacement for the 1,000-particle publication workload.

A sparse ROS2 linear solve and colored native-condensation Jacobian are now
implemented on this branch. Particle-local activity/surface formulas permit
material-wise finite differences; gas derivatives retain weighted reservoir
coupling. Unknown custom activity/surface models retain dense differences. A
regression confirms cross-particle derivatives are preserved for a deliberately
nonlocal callback. The wet-to-dry derivative perturbation now follows current
particle mass, rather than the initial wet scale. The original full-population
probe was intentionally stopped after exposing that derivative defect; its log
is diagnostic evidence, not a completed publication run.

The focused numerical subset passed 26 tests after the dry-scale fix, and the
expanded sparse suite passed all 7 tests after adding the custom-model guard.
Targeted checked-body Mypy passes for the three modified numerical modules
with imported-module errors suppressed; ordinary full-package Mypy remains a
separate gate. The full package/CUDA validation passed 10,690 tests with 91.21% coverage.
Standard Ruff/format and Mypy passed; the three new examples passed CPU replots
and the extracted CUDA download executed successfully. The 558-page website
build and rendered structure check passed. Tooling/examples excluding the
separately running native-cloud module passed 552 tests with one skip. Durable
numerical evidence is in
[the method record](../docs/contribute/validation/numerics/adaptive-methods.md).

Three new registered examples have completed full refreshes with compact data
and visually inspected figures:

- `adaptive-cloud-relaxation`: independent scalar Radau/BDF agreement, wet/dry
  reset and closed-phase budgets. ROS2 radius errors are approximately 6.05e-6
  and 6.98e-8 at relative tolerances 1e-5 and 1e-7; the coarse fixed method has
  approximately 3.70 maximum relative radius error despite closing its budget.
- `adaptive-reservoir-depletion`: independent scalar DOP853/Radau growth,
  fixed-step refinement and CPU RK23/ROS2. Tight RK23 radius error is 2.57e-8;
  vapor error normalized by initial vapor is 9.15e-8.
- `adaptive-reservoir-depletion-cuda`: the same complete comparison on the
  actual RTX 5060. CUDA RK23 errors improve from 1.82e-5 to 2.57e-8 for radius
  and 6.74e-5 to 9.15e-8 for vapor. This small case is accuracy evidence, not a
  GPU performance claim or stiff-cloud qualification.

The full single-cycle ROS2 publication workload completed, retaining 1,000
particles and the complete 60 s history. The inspected plot shows smooth dry
relaxation; peak diameter is 23.19 µm with 788 survivors. A higher-order ROS3
option now passes independent order/transient, forcing and exact restart checks.
The refreshed wet–dry comparison requires 330/1,247 accepted ROS3 steps versus
3,667/36,161 for ROS2 at the same 1e-5/1e-7 tolerances, with slightly smaller
maximum radius errors. The full single-cycle ROS3 refresh also completed,
matching the printed peak and survivor count in 299.3 s versus ROS2's 1,869.4 s.
The successful ROS2 bundle remains retained for comparison. The added method
passed 10,703 package tests with 91.21% coverage. These results do not qualify
coupled GPU stepping.

The complete ROS3 native-example suite exposed dry-inventory roundoff that
was not covered by the package tests. Both causes are now corrected: residual
refinement of the linear solve and exact conversion of unchanged scaled
coordinates. The original dry-mass assertions remain intact. All 13 native
example tests pass with the production corrections. The subsequent package
selection passes 10,710 tests on actual CUDA, and the other example/tooling
checks pass 554 tests. The unfiltered suite passes 10,766 tests with 91.41% coverage.
The numerical evidence record owns detailed commands, results and limitations.

The full single-cycle repeat completed before those corrections. The live
multi-cycle, respiratory and biomass runs were deliberately stopped because
their numerical source was superseded. The complete workloads run in a detached validation checkout
pinned to `12742e1c394c7a836d31aff3aeecbd8c0ff2b523`. The first CPU workflow and
two CPU comparison examples have completed refresh, inspection and replot.
The corrected full single-cycle chamber also completed and its figure was
inspected. The full multi-cycle chamber also completed: all three 1,000-particle
compositions, four cycles and 3,601 observations each. Its bundle/source hashes
and eight figures were verified and imported. The full respiratory publication
also completed, with all twelve species budgets intact, 201 observations and
8,921 accepted ROS3 steps. Its hashes and five figures were verified and imported.
Those earlier runs were superseded by the user's request for fully resident
CUDA execution. The biomass publication and respiratory refinement are now
complete on `9c548d9d61`, retaining full workloads and the original budgets.
The biomass turbulence model was explicitly changed to supported ST1956 shear.
The current acceptance status and remaining delivery gates are tracked in Stage 2B.


GPU cyclic/forcing ownership has been inspected. The cyclic study includes
source pulses, persistent stochastic collision state, captured fixed stepping
and byte-exact checkpoint continuation. Adaptive condensation cannot simply be
substituted into that captured graph. Resident ROS3 is now integrated with
forcing, exact same-device restart, passive-inventory preservation and actual
CUDA growth/refinement checks. The new full CUDA wet–dry comparison completed
publication, figure inspection, CPU replot and extracted-download execution.
Its errors refine from 4.07e-6 to 3.24e-8. The integrated package passed 10,788
tests with 91.30% coverage; detailed evidence belongs to the method record.
Coupled event ownership and exact restart are now implemented with explicit
state transfers. The focused coupling/numerics/archive tests pass 65 cases;
the full package gate passed 10,809 tests with 91.33% coverage. A subsequent
checkpoint-plan guard passed all 17 focused coupled tests and checked-body Mypy.
The full prepared forcing publication completed both the 6,000-cycle trajectory
and its 12,000-cycle refinement, with unchanged physical forcing. Final mass and
vapor differences are 2.32e-7 and 8.86e-9 relative. Hashes, figure inspection,
CPU replot and extracted-download import/rendering passed; the compact bundle
is retained. The workflow qualification record owns detailed measurements.
The earlier transferred cyclic-chamber run was stopped at the user's request.
Its fully resident replacement runs from `/tmp/particula-stage-02b`, pinned to
`9c548d9d61`; its complete 600 s trajectory and exact second-half continuation
passed. Plot-data serialization then failed, so an output-only correction at
`3d3753d2b` completed a fresh full publication with unchanged physics. All 31
observations, figure inspection and CPU-only replot are retained. The bounded first-interval diagnostic passed its budget
check, while later rapid evaporation required many small accepted steps.
The scientific runs, retained results and local documentation verification
are complete. GitHub publication/readiness remains pending direct push
authorization.

The respiratory time-refinement requirement is a separate numerical gate.
`scripts/assess_respiratory_refinement.py` reuses the complete published setup
and coarse solver, then compares all 201 observations with a second full run
using 100-fold tighter relative/absolute tolerances, half the maximum step and
half the output interval. Prospective consistency targets are 0.1% maximum
per-droplet relative radius difference and 0.1% phase-inventory difference
normalized separately by each species' initial total inventory. The report also
retains each material's maximum per-slot difference relative to its own initial
mass, without normalizing small solutes by water. Both trajectories must retain
the original per-species budget assertions. This is same-method consistency,
not an independent oracle or validation of the known incorrect chemical-name
matches. The completed assessment retained its compact report using this command:

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python \
  scripts/assess_respiratory_refinement.py \
  --output .artifacts/respiratory-refinement.json
```

The older CPU comparison was stopped and superseded at the user's request.
The complete CUDA-resident comparison subsequently passed all 201 observations
on revision `9c548d9d61`, with maximum relative radius difference `3.92121e-6`
and maximum per-species phase-inventory fraction `3.22563e-8`. The compact
report is `docs/data/numerics/resident-respiratory-refinement.json`; the
workflow qualification record owns the controls, work counts and evidence
limits. This respiratory acceptance item is complete.
