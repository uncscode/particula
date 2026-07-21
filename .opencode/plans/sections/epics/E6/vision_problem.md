# Vision and Problem

Particula cannot yet execute a complete fixed-shape aerosol timestep through
the current low-level GPU process set. The remaining gaps are:

1. **Dilution lacks process parity.** CPU dilution exists only as free
   functions, with no strategy/runnable reference or direct GPU equivalent.
2. **Wall loss is CPU-only.** Mature neutral and charged CPU models have no
   fixed-shape GPU implementation.
3. **Nucleation is not a supported process.** Theory exists, but there is no
   inventory-limited CPU reference or direct GPU process.
4. **Particle creation is incomplete.** Fixed GPU arrays recognize inactive
   slots but lack shared activation, diagnostics, and explicit exhaustion
   policies.
5. **Integrated evidence is missing.** No direct-call sequence demonstrates
   condensation, coagulation, dilution, wall loss, and nucleation without
   intermediate host transfers.

## Vision

Deliver trustworthy CPU references and parity-tested, low-level Warp
implementations for the missing processes. A caller will be able to run the
complete process sequence in fixed-shape, particle-resolved storage while
preserving explicit CPU/GPU transfer boundaries, caller-owned sidecars,
per-box/species conservation, and observable slot-exhaustion behavior.

## Why Now

Shipped Epic E5 supplies GPU coagulation, persistent RNG, charge handling,
inactive-slot conventions, and validation patterns; direct GPU condensation
already supplies gas coupling and reusable sidecar patterns. GPU process
completeness is the remaining prerequisite before downstream Epic G can own
backend selection and GPU-resident scheduling.
