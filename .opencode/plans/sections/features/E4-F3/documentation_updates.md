# Documentation Updates

- P1/P2 documentation is intentionally limited to the API docstrings/comments in
  `particula/gpu/kernels/condensation.py`: this is a concrete-module-only
  sidecar API. The docstrings state ownership/lifetime, active-device fp64 and
  stable-shape requirements, identity semantics, fixed-four work-versus-total
  transfer behavior, and the pre-mutation failure guarantee.
- P3 / issue #1294 made no user or roadmap documentation changes: it is
  test-only regression coverage. Test names and assertion messages label the
  `5e-2` bound as recorded case-specific stiffness evidence, not a general
  parity or conservation tolerance.
- P4 / issue #1295 updated `docs/Features/Roadmap/condensation-stiffness-study.md`
  with the shipped fixed-four behavior, case-bounded recorded evidence, transfer
  semantics, particle-only scope, and downstream E4-F4 through E4-F7 gates.
- P4 updated `docs/Features/Roadmap/data-oriented-gpu.md` with shipped E4-F3
  ownership, the package step versus concrete-sidecar distinction, scratch
  ownership/lifetime, fixed-four ordering, and bounded limitations.
- P4 added a clearly separate low-level Warp note to
  `docs/Features/condensation_strategy_system.md`, including the focused
  discoverable test command and guarded Warp/CUDA skip guidance:
  ```bash
  pytest particula/gpu/kernels/tests/condensation_test.py \
    particula/gpu/kernels/tests/condensation_stiffness_test.py -q
  ```
  Missing Warp may skip; CUDA evidence is optional when CUDA is unavailable,
  and a skip is not GPU execution.
- The P4 scope remains bounded: it does not claim CPU-strategy parity,
  `Runnable` composition, adaptive stepping, gas coupling or conservation,
  latent heat, graph capture/replay, or autodiff readiness. The recorded
  stiffness bounds apply only to the named cases, not general accuracy or
  conservation.
- No package export, second step entry point, standalone high-level API, graph
  replay, or autodiff claim was added. `warp-autodiff-limitations.md` remains
  unchanged.
