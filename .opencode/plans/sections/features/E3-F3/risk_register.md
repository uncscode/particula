# Risk Register

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| CUDA hardware is unavailable during implementation. | Benchmark evidence may be incomplete. | Record skip status, preserve optionality, and document that measured refresh requires CUDA hardware. | Implementer |
| Machine-specific timings are overgeneralized. | Users may misinterpret performance guarantees. | Include command, date, hardware context, and caveat that numbers are guidance rather than universal guarantees. | Implementer + reviewer |
| Scope expands into kernel optimization. | Feature exceeds documentation/design-decision intent and risks Epic C churn. | Keep production graph capture and parallel-within-box implementation explicitly out of scope; create follow-up only if needed. | Implementer |
| E3-F2 sampling changes alter benchmark conclusions. | Decision may be based on stale behavior. | Treat E3-F2 as a dependency and verify benchmark target reflects its selected outcome. | Implementer |
| Paired notebook docs drift from source. | Published benchmark examples may become inconsistent. | Edit paired `.py` source first and sync/validate notebook when necessary. | Documentation reviewer |
| CUDA optionality regresses. | Normal CPU-only development or CI may fail. | Keep benchmark markers and skip behavior intact; avoid mandatory CUDA assertions outside opt-in benchmarks. | Test reviewer |
