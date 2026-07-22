# Documentation Updates

## Shipped P1–P5 Status (#1395–#1399)

- No user-facing documentation files changed in P1. The concrete-only contract
  is recorded in `particula/gpu/kernels/dilution.py` and its co-located tests;
  package-level documentation remains P5 scope.
- P2 ships the code-docstring contract and package export for
  `particula.gpu.kernels.dilution_step_gpu`: it mutates only fixed-shape
  particle/gas concentrations with finite-step exponential decay and preserves
  identity/protected fields. P3 (#1397) updates the code-docstring contract and
  co-located tests for ordered full preflight, exact same-device float64 Warp
  schemas, finite/nonnegative physical values, and validation before no-op
  returns/allocation/launches. Rollback after a launched-kernel failure remains
  deferred.
- P4 (#1398) changes only `particula/gpu/kernels/tests/dilution_test.py` to add
  independent NumPy-reference Warp CPU/CUDA-optional parity and invariant
  evidence. No production API or user-facing documentation files changed.
- P5 (#1399) publishes the bounded P1–P4 direct-kernel contract in
  `docs/Features/data-containers-and-gpu-foundations.md`,
  `docs/Features/Roadmap/data-oriented-gpu.md`, and `AGENTS.md`. It reconciles
  the README and documentation indexes, and extends
  `particula/tests/dilution_docs_test.py` with hardware-free checks for scoped
  contract/deferred wording, lazy `particula.gpu.kernels` export metadata, and
  local Markdown file and anchor resolution. The documentation retains caller
  ownership, Warp-CPU `rtol=1e-12, atol=0` evidence, optional CUDA, and the
  non-bitwise/deferred-scope boundaries.

## Deferred Documentation Scope

- A runnable example must keep CPU-to-Warp and Warp-to-CPU conversion calls
  visible and outside `dilution_step_gpu`; it remains E6-F9 scope.
- Do not claim wall loss, nucleation, backend selection, scheduling, graph
  capture, autodiff, resizing, hidden fallbacks, or performance support.
