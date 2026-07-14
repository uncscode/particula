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
- Update `docs/Features/Roadmap/data-oriented-gpu.md` with E4-F3 completion,
  scratch ownership, E4-F1 refresh placement, and downstream gates.
- Update `docs/Features/Roadmap/warp-autodiff-limitations.md` if validation
  confirms static-loop replay/graph-capture claims.
- Do not add a package export, README/API page, or standalone usage example for
  P1; the sidecar is imported only from the concrete condensation module.
- Add or update a focused usage example only if the public scratch API is
  intended for users; avoid introducing a second canonical entry point.
- The plan section status and change log record P3 as shipped; P4 remains the
  planned development-documentation phase.
