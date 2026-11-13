# Dependency Map

## Inbound

- Shipped E2 fixed-shape container schemas, ownership, and conversion rules.
- Shipped direct GPU condensation and coagulation contracts.
- E6-F1/F2 dilution; E6-F3/F4 neutral and charged wall loss; E6-F5/F6
  activation and exhaustion; E6-F7/F8 CPU and direct GPU nucleation; E6-F9
  integrated sequence fixtures and documentation.
- CPU `RunnableABC` and `RunnableSequence` as reference behavior.
- Warp for Warp CPU and optional CUDA execution; NumPy CPU oracles.

## Outbound

- Epic H depends on the stable resident execution boundary for graph capture,
  profiling, and performance optimization.
- Epic I depends on stable state and execution semantics for autodiff and
  optimization.
- User examples and higher-level simulation workflows will consume E7's public
  backend-selection and session APIs.

## Sequencing

Authoritative chain:

`E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5 -> {E7-F7, E7-F8} -> E7-F9`

- E7-F1 establishes the capability model and typed boundary.
- E7-F6 freezes fallback, error, export, and stability policy before adapters.
- E7-F2, E7-F3, and E7-F4 may proceed in parallel after E7-F1 and E7-F6.
- E7-F5 integrates only the shipped adapters and resident-session contract.
- E7-F7 and E7-F8 may proceed in parallel after full scheduling exists, while
  retaining their direct dependencies shown in `child_plans.md`.
- E7-F9 is the closeout gate and depends on all E7-F1 through E7-F8 outcomes.
- Unit tests are co-located with every feature's implementation; E7-F9 adds
  cross-feature integration, documentation, and closeout evidence rather than
  deferring unit coverage.
