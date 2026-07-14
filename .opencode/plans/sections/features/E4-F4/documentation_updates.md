# Documentation Updates

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
