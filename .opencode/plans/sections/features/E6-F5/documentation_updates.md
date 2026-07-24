# Documentation Updates

- P3 remains intentionally concrete-only: `get_slot_diagnostics_gpu` is not
  package-exported. Its module and typed function docstrings document its
  caller-owned sidecars, read-only classification, and import boundary.
- P4 public-contract guidance was added to `AGENTS.md`, `docs/index.md`,
  `.opencode/guides/architecture/architecture_outline.md`, and
  `.opencode/guides/testing_guide.md`. It documents the package import,
  fixed-capacity ascending mapping, same-device `wp.float64` inputs and
  caller-owned `wp.int32` sidecars, complete preflight/no hidden transfers, and
  the no-rollback-after-writer boundary.

No user example is required until a physical particle-source process consumes
activation; E6-F9 owns the integrated direct-step example.
