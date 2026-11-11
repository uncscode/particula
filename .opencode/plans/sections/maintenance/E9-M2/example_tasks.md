# Implemented Acceptance Scenarios

1. Wrap a two-particle-species/three-gas-species triple without reconstruction;
   preserve the inert gas lane and direct current-data reads.
2. Reject individual box/species layout mismatches, then accept the compatible
   complete triple by identity. Snapshot all rejected old/candidate fields.
3. Repeated native builder builds produce separate aggregates sharing the exact
   supplied containers; no strategy-name or partitioning-count check remains.
4. Convert radius to mass and check PDF product trapezoids independently.
   PMF/PDF use V=1 and reject .25/4; those are not successful binned fixtures.
5. At resolved V=.25/1/4, verify each species inventory and number density for
   count, density and fractional weights. Controlled sampled radii give unit
   counts and preserve the old sampler/defaults.
6. Exercise all initializer exports through native aggregate construction,
   replacement and rejected tagged transfer without invoking a facade.

These scenarios are executable in the phase tests and P6 integration regression;
no scientific process or tutorial is migrated by them.
