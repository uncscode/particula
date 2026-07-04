## Open Questions

1. Should simulation `volume` remain particle-owned, become environment-owned,
   or be treated as shared simulation-domain metadata?
2. Should humidity and saturation be stored as primitive environment fields,
   derived from gas/environment state, or represented through helper methods?
3. Should vapor pressure be owned by gas state, environment state, kernel input,
   or an explicit transient transfer object?
4. Should `WarpGasData` preserve species names through side metadata, require
   caller-supplied names on CPU restoration, or intentionally remain numeric
   only?
5. What is the minimum per-box environment interface needed by condensation and
   coagulation kernels without committing to a full integrator redesign?
6. Which CPU dynamics paths should accept data containers now, and which should
   explicitly raise unsupported multi-box errors?
7. What numerical tolerances should define acceptable mass representation and
   condensation timestep behavior for downstream GPU epics?

These questions should be narrowed by E2-F1 and closed or documented by the
relevant child feature tracks.
