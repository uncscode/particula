# Risk Register

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| Import path decision exposes too broad a public API | Users depend on unstable raw Warp internals | Export/document only the two step functions unless a separate API decision approves more | Implementer |
| Quick-start implies hidden transfers | Violates Epic E3 explicit boundary guardrail | Show every `to_warp_*` and `from_warp_*` call in the example and documentation | Implementer + reviewer |
| Coagulation example conflicts with pre-`E3-F1` RNG behavior | Repeated-call examples may be misleading | Depend on `E3-F1`; otherwise demonstrate one call only and note limitation | Implementer |
| CUDA unavailable in user or CI environment | Example/test failures on common machines | Default to `device="cpu"`; make CUDA optional and skip cleanly | Implementer |
| Device mismatch examples are hard to debug | Users encounter opaque Warp errors | Add troubleshooting tied to existing validation messages and device requirements | Documentation reviewer |
| Docs example drifts from tested code | Quick-start becomes stale | Add smoke tests for no-Warp and Warp CPU paths | Implementer |

## Residual Risk

The low-level APIs remain advanced because users must manage Warp-resident data,
buffers, devices, and optional environment data explicitly. Documentation should
embrace this as a low-level contract rather than hiding it behind convenience
abstractions in this feature.
