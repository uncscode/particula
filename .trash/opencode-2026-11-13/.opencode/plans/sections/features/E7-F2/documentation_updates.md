# Documentation Updates

## Shipped P4 Architecture Update (Issue #1473)

`.opencode/guides/architecture/architecture_outline.md` now records that CPU
selected dispatch remains isothermal; selected Warp dispatch profile-preflights
before lazy resolution and forwards caller-owned `latent_heat`,
`energy_transfer`, and deferred `thermal_work` by identity. It also assigns
thermal validation/execution and exceptions to `condensation_step_gpu`, without
adding transfer, allocation, synchronization, restoration, or fallback.

## Shipped P6 Selected-Condensation Contract (Issue #1475)

`docs/Features/condensation_strategy_system.md` now publishes the bounded
selected-condensation support matrix and evidence limits: 36 declared CPU
semantic profiles, isothermal selected CPU execution, eight selected Warp
profiles, and pre-dispatch rejection for staggered or nonrepresentable/BAT Warp
mappings. It records four equal Warp substeps, per-substep vapor-pressure
refresh, P2 inventory-limited coupling, in-place primary/output mutation, the
`rtol=1e-10, atol=1e-30` integration comparison tolerance, separate inventory
checks, Warp CPU baseline, optional CUDA rows, and the E7-F4/E7-F5 handoff.

`docs/Features/data-containers-and-gpu-foundations.md` now records that the
exact ten-name public selection surface excludes selected-condensation carriers
and adapters, which remain concrete-only. It distinguishes caller-owned legacy
CPU `Aerosol`/`MassCondensation` state from resident Warp particle, gas,
environment, and sidecar state; keeps conversion, synchronization, and restore
at caller checkpoints; and states that normal Warp dispatch performs no upload,
restore, synchronization, allocation, retry, or silent CPU fallback.

The published failure narrative distinguishes write-free adapter/direct
preflight rejection from a raw-proposal failure after earlier successful
substeps, for which callers own snapshot/restore if retry is required.
`energy_transfer` remains a caller-owned write-only output and `thermal_work`
remains validated deferred state. The direct quick start remains the only
runnable explicit-transfer path; no selected-condensation import, workflow, or
public API was introduced.

## Validation

`particula/tests/execution_selection_docs_test.py` adds hardware-free Markdown
regressions for the private, non-runnable ownership subsection; support,
ownership, dispatch, and failure statements; and the new internal links. The
focused documentation test and `mkdocs build --strict` validate the rendered
contract.
