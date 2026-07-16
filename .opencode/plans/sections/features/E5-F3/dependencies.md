# Dependencies

## Upstream

- E5-F1, Mechanism Configuration and Sampling Contract, must provide stable
  canonical identifiers, resolved mechanism masks, executable-capability
  validation, additive pair-rate/majorant dispatch, and one-pass sampling.
- E5-F2, Charged Pair Physics and Charge-Conserving Merges, must provide the
  `charged_hard_sphere` scalar pair helper, charge preflight, and recipient/
  donor charge transfer and clearing. Other charged variants remain deferred
  and cannot be registered by this execution track.
- Shipped E2/E3 GPU foundations provide `WarpParticleData.charge`, fixed-shape
  fp64 arrays, bounded disjoint pair sampling, caller buffers, and persistent RNG.
- CPU charged and combined coagulation modules provide independent formula and
  additive-semantic references; they are not runtime GPU dependencies.
- NVIDIA Warp is required for Warp CPU evidence when installed. CUDA is optional
  and must skip cleanly when unavailable.

## Downstream

- E5-F6 consumes the charged rate and majorant branches when defining broader
  additive two- and four-mechanism combinations.
- E5-F7 consumes charged-only and Brownian-plus-charged fixtures in the final
  deterministic, stochastic, conservation, buffer, RNG, and device matrix.
- E5-F9 documents final user-facing support and direct examples only after
  E5-F3/F6/F7 settle the supported combination matrix.
- E5-F4 and E5-F5 are sibling mechanism tracks and may proceed independently
  after E5-F1; they should reuse, not fork, the shared sampler.

## Phase Ordering

P1 must prove and expose the charged majorant before P2 can register charged-only
execution. P2 establishes executable charged behavior and charge-conservation
evidence before P3 combines it with Brownian. P4 documents only behavior proven
by P1-P3 and remains final. Every implementation phase includes co-located tests;
there is no standalone testing phase. Classifier diagnostics: none.
