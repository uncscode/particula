# Risk Register

| Risk | Likelihood | Impact | Mitigation / Exit Evidence | Owner |
|------|------------|--------|----------------------------|-------|
| Approved charged variants remain ambiguous, causing an unbounded port | Medium | High | E5-F1/F2 freeze the subset before P2; require hard-sphere baseline and explicitly defer every non-approved CPU variant | E5-F2-P2 |
| Formula transcription or sign error reverses attraction and repulsion | Medium | Critical | Independent CPU fixtures cover same-sign, opposite-sign, zero-charge, symmetry, and published limiting behavior | E5-F2-P1/P2 |
| Exponential or reduced-property edge cases generate NaN/Inf | Medium | High | Preserve repulsive clipping/safe exponentials, add domain guards, and test mixed scales and finite boundaries | E5-F2-P1 |
| Charge buffer is malformed or on another device | Low | High | Validate shape, fp64 dtype, device, and finite values before allocation, launch, mutation, or RNG reset | E5-F2-P3 |
| Parallel merge writes race on a shared recipient | Low | Critical | Preserve the selector's disjoint-pair invariant; direct tests reject/avoid duplicate indices and document this kernel precondition | E5-F2-P4 |
| Donor charge is added but not cleared, or cleared without addition | Medium | Critical | Single merge site performs recipient add and donor clear; direct and step tests assert both plus total charge | E5-F2-P4 |
| Neutral Brownian callers regress after charge validation/launch changes | Low | High | Keep all-zero charge valid, preserve public signature/returns, and run existing Brownian/RNG/buffer regressions | E5-F2-P3/P4 |
| Documentation overclaims executable charged sampling | Medium | Medium | State that E5-F2 ships pair primitives and merge semantics only; E5-F3 owns majorants and execution | E5-F2-P5 |
| CUDA-only behavior escapes routine validation | Low | Medium | Warp CPU is mandatory when installed; parameterize optional CUDA with clean skips and identical deterministic assertions | E5-F2-P1-P4 |
