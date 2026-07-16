# Documentation Updates

- Update the `coagulation_step_gpu` docstring in
  `particula/gpu/kernels/coagulation.py` with the keyword-only configuration,
  Brownian default, additive semantics, validation order, and
  particle-resolved-only support.
- [x] Added module/class/function docstrings for the P1 configuration, resolver,
  and capability validator. They state the concrete-module-only boundary, the
  Brownian default, canonicalization, and absence of public-step integration.
- [x] Updated private helper and sampler comments/docstrings in
  `particula/gpu/kernels/coagulation.py` for P2's additive total-rate/majorant,
  one-acceptance-draw, and invalid-term skip behavior.
- Update `docs/Features/data-containers-and-gpu-foundations.md` with ownership:
  mechanism configuration is caller-owned host metadata; particle state,
  buffers, and RNG remain caller-owned Warp data.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the T1 contract,
  initial supported/reserved matrix, and links to E5-F2 through E5-F7.
- Keep `docs/Features/condensation_strategy_system.md` unchanged unless a shared
  low-level configuration convention is explicitly cross-referenced.
- Update `.opencode/plans/sections/features/E5-F1/` phase statuses and decisions
  when implementation resolves naming/export questions.
- Defer the complete direct GPU coagulation example and final support matrix to
  E5-F9 so documentation does not imply unavailable physics.

No user-facing documentation changed in Issue #1331 because the public API and
GPU runtime behavior did not change.

No user-facing documentation changed in Issue #1332: the refactor is private to
the Brownian kernel and leaves the public API unchanged.
