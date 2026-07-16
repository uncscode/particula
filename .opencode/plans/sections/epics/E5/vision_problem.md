# Vision and Problem

Particula's low-level particle-resolved GPU coagulation path currently supports
Brownian collisions, while the CPU system also supports charged,
sedimentation, turbulent-shear, and additive combined kernels. This leaves
GPU users unable to run the broader physics set despite the shipped GPU data,
validation, RNG, and fixed-buffer foundations.

The current state creates five program-level problems:

1. **Physics coverage is incomplete.** Charged, SP2016 sedimentation, ST1956
   turbulent-shear, and combined mechanisms are unavailable on the GPU path.
2. **Charge is stored but not honored.** `WarpParticleData` carries charge,
   but collision rates do not use it and merges do not transfer and clear it.
3. **Combination semantics are undefined.** Independent stochastic passes
   could double-count opportunities; supported combinations need one additive
   pair-rate calculation and one bounded sampling pass.
4. **Evidence and boundaries are incomplete.** Pair parity, stochastic bounds,
   mass and charge conservation, device expectations, and unsupported modes
   need explicit tests and documentation.
5. **A prior documentation obligation remains open.** The independent CPU/Warp
   condensation parity walkthrough and deferred-capability ownership record
   carried from E4 still need publication.

## The Vision

After E5 ships, particle-resolved users can select validated charged,
Brownian-plus-charged, sedimentation, turbulent-shear, and supported additive
combined GPU coagulation mechanisms through the existing low-level execution
model. Pair physics is traceable to CPU references, merges conserve species
mass and charge, persistent RNG and fixed-shape buffer ownership remain stable,
and support limits are explicit. The E4 condensation documentation obligation
is also closed with reproducible, independently constructed CPU and Warp
evidence.

## Why Now

E2-E4 established stable GPU schemas, explicit transfer boundaries,
device-aware validation, persistent RNG state, and the direct condensation
contract. Issue #1320 identifies this as the next ordered roadmap effort before
GPU process completeness and higher-level GPU-resident simulation.
