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

**Delivered increments:** E5-F2-P1 (#1336) supplies the internal scalar fp64
pair-property foundation and independent co-located parity tests. E5-F2-P2
(#1337) adds the internal scalar `charged_hard_sphere_wp` composite rate helper
and independent deterministic parity evidence. E5-F2-P4 (#1339) updates the
private merge kernel so each accepted pair transfers signed fp64 donor charge
to its recipient and clears the donor with mass and concentration. Direct
kernel and step-level conservation coverage now prove that state transition;
public exports, data containers, Brownian selection, the public step API,
return tuple, sidecars, and RNG ownership remain unchanged. Charge preflight,
public/model dispatch, and charged sampling remain separate phases.

This is epic-linked track T2 under E5. Classifier diagnostics: none.
