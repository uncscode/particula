# Documentation Updates

- P1 documentation is intentionally limited to the API docstrings in
  `particula/gpu/kernels/condensation.py`: this is a concrete-module-only
  sidecar API. The docstrings state ownership/lifetime, active-device fp64 and
  stable-shape requirements, identity semantics, one-update P1 behavior, and
  the pre-mutation failure guarantee.
- Update `docs/Features/Roadmap/condensation-stiffness-study.md` to mark
  fixed-count four-substep integration as production behavior, document total
  transfer semantics, and retain recorded-grid evidence and caveats.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with E4-F3 completion,
  scratch ownership, E4-F1 refresh placement, and downstream gates.
- Update `docs/Features/Roadmap/warp-autodiff-limitations.md` if validation
  confirms static-loop replay/graph-capture claims.
- Do not add a package export, README/API page, or standalone usage example for
  P1; the sidecar is imported only from the concrete condensation module.
- Add or update a focused usage example only if the public scratch API is
  intended for users; avoid introducing a second canonical entry point.
- Mark this plan's phases and change log as shipped during final documentation.
