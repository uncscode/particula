# Documentation Updates

- [x] Updated the `coagulation_step_gpu` docstring in
  `particula/gpu/kernels/coagulation.py` with the keyword-only configuration,
  Brownian default, resolved-mask dispatch, validation order, exact wrong-type
  error, and particle-resolved-only support.
- [x] Added module/class/function docstrings for the P1 configuration, resolver,
  and capability validator. They state the concrete-module-only boundary, the
  Brownian default, canonicalization, and absence of public-step integration.
- [x] Updated private helper and sampler comments/docstrings in
  `particula/gpu/kernels/coagulation.py` for P2's additive total-rate/majorant,
  one-acceptance-draw, and invalid-term skip behavior.
- [x] Issue #1334 published the canonical host-configuration and Warp-sidecar
  ownership contract in `docs/Features/data-containers-and-gpu-foundations.md`:
  configuration is host metadata; particle state, collision outputs, and RNG
  remain caller-owned same-device Warp data.
- [x] Issue #1334 published the identical import split, particle-resolved-only
  boundary, executable/reserved matrix, and one-pass extension contract in
  `docs/Features/Roadmap/data-oriented-gpu.md`.
- Keep `docs/Features/condensation_strategy_system.md` unchanged unless a shared
  low-level configuration convention is explicitly cross-referenced.
- [x] Issue #1334 updated the P4 phase status and success criteria without
   changing container schemas or GPU runtime behavior.
- No focused test, import-smoke, or documentation-build validation is recorded
  for this documentation-only closeout.
- [x] The complete direct GPU coagulation example and final support matrix remain
  deferred to E5-F9 so documentation does not imply unavailable physics.

No user-facing documentation changed in Issue #1331 because the public API and
GPU runtime behavior did not change.

No user-facing documentation changed in Issue #1332: the refactor is private to
the Brownian kernel and leaves the public API unchanged.

Issue #1333 updated the low-level public-step docstring plus `readme.md` and
`AGENTS.md`. Issue #1334 completed P4's developer contract publication; the
complete end-user example and final support matrix remain E5-F9 work.
