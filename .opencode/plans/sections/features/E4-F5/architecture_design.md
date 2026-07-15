# Architecture Design

Issues #1302 and #1303 establish P1 gating and private P2 inventory
finalization. Issue #1304 completes public orchestration in
`condensation_step_gpu()` without changing its API or exports.

After aggregate preflight validates primary physical state, sidecar metadata,
and ownership, supplied scratch is resolved once and omitted P2 workspace is
allocated once. Only then are the caller-owned total and optional energy output
cleared. This boundary precedes normalization, vapor-pressure refresh, and all
mutable launches.

The public call always executes four equal substeps. Each cycle computes and
P1-gates a raw proposal, validates that fresh output for finiteness, uses P2 to
bound evaporation and scale uptake from gas plus same-substep release, then
applies the finalized transfer exactly once to particle mass. Private Warp
kernels add that finalized transfer to the whole-call total and, in ascending
particle-index order, subtract its concentration-weighted aggregate from
`gas.concentration`. Thus each later mass-transfer proposal reads gas coupled
by the preceding completed cycle. Vapor-pressure refresh does not read gas
concentration.

The returned total and optional energy output use only the P2-finalized
whole-call transfer; raw work remains caller-owned final P1 proposal storage.
Stale work contents are allowed because it is output-only. A nonfinite fresh
proposal fails before P2, particle, gas, total, or energy mutation for its
cycle. Aggregate preflight is atomic, but there is intentionally no whole-call
rollback for a later-cycle proposal failure: completed earlier cycles remain
applied and coupled, and the failing cycle may have written raw work only.
