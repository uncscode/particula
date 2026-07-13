# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| E4-F1 refresh is placed outside the loop, producing stale substep physics | Medium | High | Launch-count/order tests assert refresh before each transfer | E4-F3 implementer |
| Invalid later scratch input is discovered after an earlier buffer is cleared | Medium | High | Validate the full input set before any write or launch; snapshot rejection tests | E4-F3 implementer |
| API return changes from total-call to final-substep transfer | Medium | High | Preserve prototype accumulator semantics and explicit tests/docs | E4-F3 implementer |
| Scratch inputs complicate compatibility or positional calls | Medium | Medium | Add keyword-only options and retain existing public export/call forms | API reviewer |
| Default-path allocations undermine graph/repeated-step goals | Medium | Medium | Document defaults; instrument all-scratch path for zero required allocation | Performance reviewer |
| Recorded `5e-2` evidence is misrepresented as universal accuracy | Low | Medium | Label it as fixture-specific evidence in tests and docs | Documentation owner |
| E4-F2 assumptions leak into this parallel track | Low | Medium | Define a refresh hook/location but depend only on E4-F1 | E4 technical lead |
| CUDA-specific device behavior diverges | Medium | Medium | Require Warp CPU tests and optional clean-skipping CUDA parity | Test owner |
| Static-loop graph/autodiff claims exceed evidence | Medium | Medium | Keep claims conditional until replay/capture validation in downstream E4-F6 | E4 technical lead |
