# Review Decisions and Remaining Acceptance

The user approved this combined six-phase implementation plan and selected:

- Explicit temporary legacy aggregate/builder modules instead of a native
  atmosphere facade or postponed public export switch.
- Small initializer functions alongside existing native data builders.
- Structural validation followed by ParticleData.validate_representation for
  native construction and every replacement, including same-object assignment.
- Lossless scalar/singleton/redundant shape normalization; reject differing
  values instead of reproducing legacy first-row/column truncation.

M1's delivered contract governs: PMF/PDF are densities at fixed V=1; only
resolved density input multiplies by physical V. Samples retain unit counts,
with modal concentrations determining mixture weights only. Existing RNG
behavior remains unchanged. Tagged Warp upload remains rejected.

Implementation, capability and temporary consumer decisions are resolved.
The final validation record owns remaining check status and the explicit Kyle
Gorkowski M2 acceptance required before M3 admission. Approval of the plan is
not a fabricated final handoff approval, merge or release.
