# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Compatibility signature omits a structural binding and permits stale replay | Medium | Critical | Enumerate session primaries, all resource manifests, request configurations, diagnostics, communication, schedule, and RNG identities; parametrize one drift test per field | E8-F1 P1/P3 implementer |
| Signature is too strict and invalidates ordinary payload or active-slot changes | Medium | High | Separate identity/schema fields from mutable payload values; test active/free slot, concentration, and RNG-word evolution as compatible | E8-F1 P1 implementer |
| Lifecycle diverges from resident session fault/terminal semantics | Medium | High | Reuse `ResidentStepGuard` and explicit writer-failure classification; prohibit independent recovery, rollback, and retry states | E8-F1 P2 implementer |
| Warp CPU or unavailable CUDA is mistaken for successful capture support | Medium | High | Use explicit capability outcomes and CUDA-marked clean skips; never select CPU fallback | E8-F1 P1 implementer |
| Host validation, allocation, or synchronization leaks into future replay | High | High | Keep E8-F1 operations metadata-only and make setup/replay responsibilities explicit for E8-F2; add no-hidden-work spy tests | E8-F1/E8-F2 owners |
| Automatic recapture masks invalid state and changes RNG trajectory | Medium | High | Require explicit retirement and fresh capture creation; never initialize/reset streams during compatibility or recapture checks | E8-F1 P2/P3 implementer |
| Native graph handle is serialized or restored across incompatible devices | Low | Critical | Exclude graph handles from checkpoints; bind to exact device and lifecycle; require fresh capture after restart | E8-F1 and checkpoint owners |
| Contract feature is misdocumented as complete captured-loop support | Medium | Medium | Scope docs and success criteria explicitly defer execution parity, example, benchmark, and profiling claims to E8-F4 through E8-F8 | E8-F1 P4 implementer |
| Optional CUDA coverage is unavailable in CI | High | Medium | Keep hardware-free lifecycle/signature unit tests mandatory; make CUDA rows pass-or-clean-skip evidence without inferring success | Test owner |
| Contract layer becomes a second scheduler | Low | High | Retain exact resolver-produced graph/schedule and delegate authoritative ordering to `ResidentSimulationScheduler` | Architecture reviewer |
