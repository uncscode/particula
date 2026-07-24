# Documentation Updates

- P3 requires no user-facing documentation: `get_slot_diagnostics_gpu` is an
  intentionally unexported direct-Warp concrete-module API. Its module and
  typed function docstrings document its caller-owned sidecars, read-only
  classification, and import boundary.
- P4/P5 remain responsible for documentation of activation, downstream capacity
  handling, and any user-facing or developer-guide contract once that surface
  exists.

No user example is required until a physical particle-source process consumes
activation; E6-F9 owns the integrated direct-step example.
