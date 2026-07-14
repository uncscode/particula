# Documentation Updates

## P4 implementation record (issue #1300)

Issue #1300 updated `docs/Features/Roadmap/data-oriented-gpu.md`,
`docs/Features/data-containers-and-gpu-foundations.md`, and
`docs/Features/condensation_strategy_system.md`. The documents now record the
shipped per-species latent-rate correction, issue #1272 signed diagnostic
`Q[box, species] = sum_particles(Delta m_applied) * L[species]`, units and
signs, optional caller-owned fp64 sidecars, overwrite-after-preflight behavior,
and the exact omitted/zero-latent isothermal fallback. They also publish the
required Warp-CPU and availability-skipped CUDA commands and retain the stated
non-goals (including temperature evolution and gas coupling).

- P2 updated the user-facing `docs/index.md` and
  `docs/Features/condensation_strategy_system.md` to document the optional
  per-species correction in four fixed substeps, exact isothermal behavior for
  omitted/zero latent heat, CPU-oracle/Warp parity, and the bounded contract.
- Production docstrings now state that latent heat is applied per fixed
  substep and that `thermal_work` is validated but deferred P3 state.
- Guidance in `docs/Theory/Technical/Dynamics/Condensation_Equations.md` and
  the GPU roadmap record the shipped correction/parity boundary; temperature
  feedback and gas coupling/conservation remain deferred.
- The user-facing contract remains bounded: no temperature evolution, gas
  mutation/conservation claim, adaptive stepping, high-level runnable, graph
  capture/replay, autodiff guarantee, or hidden host transfer.
