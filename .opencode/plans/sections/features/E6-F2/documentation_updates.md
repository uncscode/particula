# Documentation Updates

## Shipped P1/P2 Status (#1395, #1396)

- No user-facing documentation files changed in P1. The concrete-only contract
  is recorded in `particula/gpu/kernels/dilution.py` and its co-located tests;
  package-level documentation remains P5 scope.
- P2 ships the code-docstring contract and package export for
  `particula.gpu.kernels.dilution_step_gpu`: it mutates only fixed-shape
  particle/gas concentrations with finite-step exponential decay and preserves
  identity/protected fields. P3 retains coefficient-value validation, complete
  preflight, and rollback.

## Future P5 Updates

- Update `docs/Features/data-containers-and-gpu-foundations.md` with field
  ownership: dilution mutates particle/gas concentration only, while transfers
  remain explicit caller actions.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with E6-F2/T2 status,
  supported scalar/per-box direct inputs, Warp CPU parity evidence, and deferred
  Epic G orchestration.
- Update `AGENTS.md` with the canonical direct import, input shapes, exact no-op
  behavior, atomic validation, protected fields, and focused test command.
- Cross-reference E6-F1 as the CPU oracle and E6-F9 as the future integrated
  direct-call consumer; do not claim wall loss, nucleation, backend selection,
  scheduling, graph capture, or performance support.
- If a runnable example is added, keep CPU-to-Warp and Warp-to-CPU conversion
  calls visible and outside `dilution_step_gpu`; default to Warp CPU and make
  CUDA optional. A broader integrated example remains E6-F9 scope.
- Update these plan sections with final API names, phase issue numbers, recorded
  tolerances, focused commands, and shipped statuses as implementation lands.
