# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the validated
  three-way matrix, supported scenarios, tolerances, and explicit evidence gaps.
- Update `docs/Features/data-containers-and-gpu-foundations.md` only if the
  validation clarifies ownership, synchronization, diagnostics, or RNG rules;
  do not publish graph-capture internals as public APIs.
- Update `.opencode/guides/testing_guide.md` with the final focused command matrix
  only when it adds durable guidance beyond the existing GPU policy.
- Update `AGENTS.md` with canonical commands and pass-or-clean-skip CUDA behavior
  after the implementation is executable.
- Update E8 and E8-F5 plan sections with shipped phase status, exact commands,
  literal outcomes, unavailable hardware rows, and downstream handoffs.
- Issue #1575 changed no user documentation: its implementation is test-only and
  does not alter production APIs, modules, or architecture.
- Issue #1576 likewise changed no user documentation: it adds test-only
  uncaptured Warp-CPU parity/conservation and no-work evidence in
  `captured_full_loop_test.py`, with no production or API change.
- Issue #1577 likewise changed no user documentation: it adds optional,
  test-only native-CUDA capture/replay parity and diagnostic evidence in
  `particula/execution/tests/captured_full_loop_test.py`, with no production,
  API, export, architecture, or example change.
- Issue #1578 changed no user documentation: it adds test-only resident RNG
  lifecycle/checkpoint-continuation and graph-replay-rejection evidence, with
  no production, API, export, architecture, example, or user-doc change.
- Do not add a new user example in this feature; the sibling graph-capture
  example track owns runnable usage and limitations.
- Validate all changed documentation with contract tests and
  `mkdocs build --strict`.

## P5 Reconciliation (#1579)

The roadmap, hardware-free regression test, and testing guide were updated. The
testing guide now includes the graph-capture documentation regression alongside
the existing hardware-free GPU documentation check, with `--no-cov` to preserve
its assertion-only status. The phase inventory is therefore the roadmap,
testing guide, graph-capture documentation test, and E8-F5 plan records. The
dedicated AGENTS block remains unchanged because strict MkDocs verification is
unavailable. The final strict build is the sole shipment blocker.
