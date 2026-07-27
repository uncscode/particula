# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| E7-F1 or E7-F6 contracts change after adapter work starts | High | Medium | Gate implementation on shipped upstream contracts; keep adapter registration narrow and typed | E7-F2 owner |
| CPU and Warp condensation algorithms are treated as identical | High | Medium | Publish supported mappings and per-case tolerances; use independent references and avoid bitwise claims | Scientific reviewer |
| Adapter validation changes kernel ordering or failure semantics | High | Medium | Keep selection validation separate; delegate detailed schema/value checks; lock call and mutation order in tests | E7-F2 owner |
| Hidden conversion, synchronization, or fallback enters convenience API | High | Medium | Transfer spies, unavailable-device tests, E7-F6 errors, and explicit ownership documentation | API reviewer |
| Sidecar aliases, wrong devices, or unstable shapes corrupt resident state | High | Low | Reuse existing validators, preserve exact identities, reject metadata/alias conflicts before launch | GPU maintainer |
| Unsupported staggered/BAT configuration is approximated silently | High | Low | Fail closed through capability declarations with dedicated negative tests | Condensation maintainer |
| Result normalization hides gas/energy mutation or partial commits | Medium | Medium | Record mutation metadata explicitly and document preflight versus post-launch boundaries | E7-F2 owner |
| CUDA-only behavior escapes routine validation | Medium | Low | Require Warp CPU baseline; keep optional CUDA rows and clean skips | Test owner |
| Public exports accidentally stabilize concrete GPU internals | Medium | Medium | Follow E7-F6 export policy and extend exact-export regressions | API reviewer |
| Scope expands into sessions, scheduling, performance, or autodiff | Medium | Medium | Enforce E7-F4/F5, Epic H, and Epic I boundaries during review | Epic E7 owner |
