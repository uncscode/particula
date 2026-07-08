# Child Plans

## Feature Tracks

| ID | Feature Plan | Status | Notes |
|----|--------------|--------|-------|
| E3-F1 | Persist coagulation RNG state | Draft | Seed once and preserve caller-owned RNG state across repeated `coagulation_step_gpu` calls. |
| E3-F2 | Harden or characterize mixed-scale coagulation sampling | Draft | Mixed NPF/droplet rejection-sampling behavior; depends on E3-F1. |
| E3-F3 | Record one-thread-per-box coagulation decision | Draft | Decision record and scaling limits; depends on E3-F2. |
| E3-F4 | Resolve low-level kernel exports and quick-start | Draft | Direct condensation/coagulation GPU quick-start and import-path decision; depends on E3-F1. |
| E3-F5 | Formalize device-aware pytest policy | Draft | Warp CPU, CUDA-if-available, parity, and stochastic tolerance policy; depends on E3-F1 and E3-F2. |
| E3-F6 | Add runnable `CondensationLatentHeat` docs example | Draft | CPU docs example deferred from Epic B/E1. |
| E3-F7 | Add CPU latent-heat conservation baseline | Draft | Integration-level CPU baseline for future Epic D GPU parity; depends on E3-F6. |

## Maintenance Tracks

Maintenance Tracks: none

## Research Tracks

Research Tracks: none

## Notes

The epic plan system rejected `add-phase` for epic plans with `Plan type 'epic'
does not support phases`; therefore the executable work breakdown is captured
as feature tracks and milestones rather than plan phases.
