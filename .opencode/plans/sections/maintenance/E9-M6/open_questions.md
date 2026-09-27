# Open Questions

1. [ ] Which exact M1–M5 revisions and named reviewers supply the accepted
   completed-and-validated entry packet? Drafted sections are not evidence.
   Resolve before P1; include final API spelling, species/normalization decisions
   and the entire temporary seam inventory.
2. [ ] Which legacy-named representation conversion utilities or builder
   capabilities remain scientifically useful after M2? P1 must reconcile
   `change_particle_representation.py` and all preset rows against real native
   consumers; preserve enduring behavior, delete only obsolete facade machinery.
3. [ ] What is the final approved M5 supported example/pair inventory and narrow
   historical exception list? Freeze every path and execution requirement before
   deletion; do not silently reclassify failing supported tutorials as historical.
4. [ ] Who signs final v0.3.0 readiness and where should its evidence be
   published? Recommendation: the proposed
   `docs/Features/Roadmap/data-native-v030-closeout.md`, with PR artifact links
   and explicit E9 maintainer approval. Release publication remains separate.
5. [ ] Which environment provides the required installed-Warp baseline and all
   supported CPU/notebook dependencies? Assign execution ownership before P5/P6.
   CUDA remains optional pass-or-clean-skip; unavailable required runs block
   readiness rather than broadening fallback or weakening checks.

Already settled by issue #1602: removal is LAST; all tracks are serial;
ParticleRepresentation/GasSpecies/Atmosphere are removed, not permanently
shimmed; enduring scientific tests and CPU-Warp helpers remain; CPU execution
is single-box; there is no new GPU API or arbitrary coverage gate. These are
constraints, not open implementation choices.
