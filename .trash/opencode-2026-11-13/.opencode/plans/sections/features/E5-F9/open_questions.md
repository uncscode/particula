# Open Questions

- [x] Which final E5-F7 validation report path and E5-F1-F6 artifact anchors are
  stable at P3 start?
  - Resolved 2026-07-16: the report target is
    `docs/Features/Roadmap/coagulation-validation.md`. P3 links only shipped,
    resolving implementation/test paths recorded by E5-F1 through E5-F6,
    including `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/dynamics/coagulation_funcs.py`, and the canonical E5-F7 test
    modules. Provisional headings or issue comments are not artifact anchors.
- [x] Should the example show one approved additive row or the smallest single
  mechanism plus configuration alternatives?
  - Resolved 2026-07-16: show the smallest deterministic supported single-
    mechanism row, Brownian, with explicit transfers, buffers, and persistent
    RNG. List approved alternative mechanism sets in prose and link the
    validation report rather than turning the example into a test matrix.
- [x] How is the P4 gate represented when child issue numbers are assigned?
  - Resolved 2026-07-16: require authoritative feature/phase status, issue
    references, resolving artifact paths, focused command results, and
    required Warp CPU evidence. Record CUDA as pass/skip/not-run. Issue closure
    alone never satisfies the gate.
- [x] Are classifier diagnostics present?
  - Resolved 2026-07-15: none; preserve this in the completion record.
