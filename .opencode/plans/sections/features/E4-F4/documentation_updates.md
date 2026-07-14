# Documentation Updates

## P3 implementation record (issue #1299)

The production `condensation_step_gpu()` docstring now documents the
caller-owned, write-only `energy_transfer` contract: active-device fp64
`(n_boxes, n_species)` output, overwrite-after-preflight semantics, valid
stale NaN/Inf storage, no third return item, and no hidden host transfer. No
user-facing documentation files changed for P3 because issue #1299 explicitly
scoped P4 documentation work out.

- P2 updated the user-facing `docs/index.md` and
  `docs/Features/condensation_strategy_system.md` to document the optional
  per-species correction in four fixed substeps, exact isothermal behavior for
  omitted/zero latent heat, CPU-oracle/Warp parity, and the bounded contract.
- Production docstrings now state that latent heat is applied per fixed
  substep and that `thermal_work` is validated but deferred P3 state.
- Guidance in `docs/Theory/Technical/Dynamics/Condensation_Equations.md` and
  the GPU roadmap in `docs/Features/Roadmap/data-oriented-gpu.md` now record
  the shipped correction/parity boundary and defer temperature feedback, gas
  coupling/conservation, and energy bookkeeping.
- The user-facing contract remains bounded: no temperature evolution, gas
  mutation/conservation claim, energy diagnostic, or hidden host transfer;
  signed energy accounting remains P3/P4 scope.
