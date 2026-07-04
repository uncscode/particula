# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Changing missing-name behavior breaks callers relying on placeholders. | Medium | Medium | Add compatibility tests, document migration path, consider warning-before-error if needed. | E2-F4 implementer |
| Vapor pressure is assigned to the wrong owner, conflicting with environment tracks. | High | Medium | Align with `E2-F2` and `E2-F3`; prefer explicit GPU transfer buffer unless ownership is resolved otherwise. | E2-F4 implementer |
| Silent zero vapor pressure remains and produces misleading GPU condensation results. | High | Medium | Test missing-input behavior explicitly and document when zeros are intentional. | E2-F4 implementer |
| Warp limitations prevent storing string metadata directly. | Medium | High | Treat names as CPU metadata or sidecar data; do not attempt string fields in Warp structs. | E2-F4 implementer |
| Non-binary GPU partitioning values cast to surprising CPU booleans. | Medium | Low | Decide and test validation or casting semantics during P2. | E2-F4 implementer |
| Optional Warp dependency makes tests flaky in environments without Warp. | Medium | Low | Reuse `pytest.importorskip("warp")` and CPU device tests. | E2-F4 implementer |
