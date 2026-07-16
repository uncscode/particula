# Overview

- **Problem Statement:** Parent epic E5 stores per-particle charge in
  `WarpParticleData`, but the Brownian-only GPU coagulation path neither exposes
  approved charged pair-rate primitives nor transfers charge when applying an
  accepted merge. The current merge therefore conserves species mass while
  silently leaving donor charge in an inactive slot.
- **Value Proposition:** E5-F2 supplies deterministic fp64 Warp pair-physics
  building blocks and makes every low-level coagulation merge conserve and
  relocate charge. This creates the physics and state-mutation foundation that
  E5-F3 can safely connect to one-pass charged sampling.
- **User Stories:**
  - As a GPU physics developer, I want charged pair helpers to match independent
    CPU references so I can extend the sampler without duplicating formulas.
  - As a simulation user, I want recipient charge to equal the sum of both
    colliders and donor charge to clear so inactive slots cannot retain state.
  - As a maintainer, I want invalid charge arrays rejected before launch so
    particle state and caller-owned RNG remain unchanged on failure.

This is epic-linked track T2 under E5. Classifier diagnostics: none.
