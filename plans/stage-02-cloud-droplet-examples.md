# Cloud and droplet integration follow-through

Status: in progress. Added to the active Stage 2A/2B objective by the user after
core implementation and validation. The full goal is not complete until this
work is implemented, executed and reviewed. Continue in the existing Stage 2B
worktree and dependent draft [PR #1660](https://github.com/uncscode/particula/pull/1660).

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
checks pass 554 tests. Unfiltered coverage validation remains in progress.
The numerical evidence record owns detailed commands, results and limitations.

The full single-cycle repeat completed before those corrections. The live
multi-cycle, respiratory and biomass runs were deliberately stopped because
their numerical source was superseded. Refresh all four complete workloads,
plus the affected comparison examples, from a fixed revision before accepting
the migration. Keep the library unchanged during each refresh; the publication
hash guard rejects a changed runtime. Current process records are retained in
`.artifacts/cloud-followthrough-runs.json`; verify live handles before resuming
or replacing a run.

GPU cyclic/forcing ownership has been inspected. The cyclic study includes
source pulses, persistent stochastic collision state, captured fixed stepping
and byte-exact checkpoint continuation. Adaptive condensation cannot simply be
substituted into that captured graph. A resident ROS3 prototype passes an
independent growth/refinement check on real CUDA, but is not yet part of the
runtime. Complete forcing/restart support and coupled event ownership, then run
the full cyclic chamber and prepared forcing workloads. The small CUDA examples
do not replace that work. Full results, final narratives, review and PR readiness
remain incomplete; no completion or merge is claimed.
