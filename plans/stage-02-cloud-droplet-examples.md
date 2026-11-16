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
not yet qualify the example migrations above. A full-population single-cycle
condensation probe is running to establish whether RK23 resolves the dry-phase
stiffness at practical cost before selecting an example policy. No speed or
stability improvement is claimed from that pending probe.
