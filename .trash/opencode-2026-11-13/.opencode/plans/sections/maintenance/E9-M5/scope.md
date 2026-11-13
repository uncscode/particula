# Scope

## Modules and directories

- All published `docs/Examples/` entrypoints, topic indexes, paired Python and
  notebook sources, and their navigation links. The appendix seeds an inventory;
  P1 reconciles it against the entire tree and `mkdocs.yml`, not just grep hits.
- Foundational Aerosol, Gas_Phase, Particle_Phase and Data_Containers tutorials;
  Dynamics process/customization examples, Chamber_Wall_Loss, Nucleation,
  Simulations, and Activity/Equilibria examples where state construction appears.
- `docs/Features/particle-data-migration/`,
  `docs/Features/data-containers-and-gpu-foundations.md`, affected process/API
  guidance, generated-reference inputs, relevant readme/quick-start snippets
  and contributor guidance. Update links when legacy tutorial names change.
- Existing documentation/example regression suites in `particula/tests/`,
  `particula/gpu/tests/`, `particula/execution/tests/`; new adjacent
  `*_test.py` fixtures only where an existing suite cannot express the contract.

## Interfaces and APIs

Publish approved M1 units/species alignment/access semantics, M2 native
construction/presets and identity-preserving replacement, M3 process-owned
physics and M4 composed CPU workflows. Show whole-container access, individual
replacement, coordinated replacement and rejection without partial publication.
Reconcile existing direct-Warp and resident examples without expanding APIs.

## Out of scope

No production physics changes, new accessors, redesigned construction, legacy
class/bridge/export deletion, permanent compatibility facade, version bump or
release approval. M6 owns deletion and final release readiness. Missing native
capabilities reopen upstream gates rather than prompting local workarounds.
Document upstream approved distribution metadata and normalization corrections;
M5 does not implement them. No general multi-box CPU execution, further
precision/schema changes, hidden transfers,
GPU fallback/rebinding, graph-capture redesign, performance claims or Epic I.
Do not retire a supported tutorial merely to avoid migrating its scientific
content. An unchanged example still needs an explicit inventory disposition.
