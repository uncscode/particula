# Scope

E5-F1 defines and integrates the configuration and sampling contracts needed
to extend the existing low-level GPU coagulation step. It keeps Brownian calls
source-compatible while making future mechanism terms additive and the
particle-resolved support boundary executable rather than implicit.

## In Scope

- Add a frozen, concrete-module mechanism configuration with a canonical,
  non-empty, duplicate-free mechanism tuple and explicit distribution type.
- Preserve `mechanisms=None` as the legacy Brownian particle-resolved path.
- Reserve stable identifiers for `brownian`, approved charged physics,
  `sedimentation_sp2016`, and `turbulent_shear_st1956` without claiming that
  downstream physics implementations already execute.
- Separate structural configuration validation from the executable capability
  check, with clear ownership for extending the capability matrix.
- Refactor the Brownian path behind pair-rate and safe-majorant interfaces that
  sum enabled terms before one candidate/acceptance pass.
- Define one-pass semantics: one candidate stream, one total majorant, one
  acceptance decision, one collision-pair buffer, and one RNG-state update.
- Reject empty, duplicate, unknown, unsupported-combination, and
  non-particle-resolved requests before allocation, launch, particle mutation,
  collision-buffer clearing, or RNG advancement.
- Add co-located unit and integration tests for compatibility, additive
  dispatch semantics, and state-preserving failures.

## Out of Scope

- Charged formulas, charge-conserving merge implementation, or charged
  stochastic execution (E5-F2 and E5-F3).
- Settling physics and SP2016 execution (E5-F4).
- Dissipation/fluid-density inputs and ST1956 execution (E5-F5).
- Enabling multi-mechanism production combinations or proving their total
  majorants (E5-F6).
- The cross-mechanism evidence matrix (E5-F7), E4 condensation walkthrough
  (E5-F8), and final user example/roadmap closeout (E5-F9).
- Binned or continuous-PDF GPU coagulation, high-level `Runnable` integration,
  dynamic slots, hidden CPU fallback/transfers, graph capture, broad autodiff,
  adaptive stepping, or performance redesign.
