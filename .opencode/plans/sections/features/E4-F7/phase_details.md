# Phase Details

Phase issue creation is intentionally deferred until E4 implementation issues
are generated and scheduled; `TBD` is not an unresolved design decision.

- [x] **E4-F7-P1:** Publish canonical GPU condensation support contract and guardrails
  - Issue: #1314 | Size: S | Status: Completed
  - Completed: 2026-07-15
  - Delivered: Published the bounded direct-kernel contract in `docs/Features/data-containers-and-gpu-foundations.md`; aligned `docs/Features/particle-data-migration.md` to the required `thermodynamics=` and two-item/in-place contract; added text-only publication regressions.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/particle-data-migration.md`, `particula/tests/condensation_latent_heat_docs_test.py`
  - Tests: `pytest particula/tests/condensation_latent_heat_docs_test.py -q -Werror`.

- [x] **E4-F7-P2:** Extend the explicit-transfer condensation example with regression tests
  - Issue: #1315 | Size: S | Status: Completed
  - Completed: 2026-07-15
  - Delivered: Reworked the quick-start into the supported gas-coupled condensation-only path with lazy Warp/direct/concrete imports, explicit CPU↔Warp conversion and final restore, and two calls reusing the same complete fp64 scratch, latent-heat, and energy sidecars. The gallery entry now describes those bounded guarantees.
  - Files: `docs/Examples/gpu_direct_kernels_quick_start.py`, `docs/Examples/index.md`, `particula/gpu/tests/gpu_direct_kernels_example_test.py`
  - Tests: Exact output and forced no-Warp import isolation; lazy public/concrete import contract; mocked explicit conversion, restore, sidecar identity, and failure behavior; plus guarded real Warp-CPU transfer/energy/gas-coupling checks. `pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q -Werror`.

- [x] **E4-F7-P3:** Publish troubleshooting and focused reproduction commands with documentation tests
  - Issue: #1316 | Size: S | Status: Completed
  - Completed: 2026-07-15 | Commit: `cb05def58`
  - Delivered: Added deterministic direct-condensation troubleshooting and the canonical focused reproduction matrix to the foundations guide; added one migration pointer and one README discovery link. The matrix uses Warp `device="cpu"` as the required baseline and labels CUDA as optional/local additive evidence.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/particle-data-migration.md`, `readme.md`, `particula/tests/condensation_latent_heat_docs_test.py`
  - Tests: Scoped text-only command/path, heading, link, constraints, and evidence-separation assertions run without Warp/CUDA imports; published focused suites retain `-Werror` where advertised.

- [x] **E4-F7-P4:** Update development documentation and roadmap status
  - Issue: #1317 | Size: XS | Status: Completed
  - Completed: 2026-07-15
  - Delivered: Added distinct canonical-contract discovery links in the gallery
    and README, published `Shipped | E4` after evidence verification, and added
    text-only link, boundary, and roadmap-status guards.
  - Prerequisites verified: E4-F1 through E4-F6 `phase_details.md` records;
    E4-F7-P1 #1314, P2 #1315, and P3 #1316 records in this section.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Examples/index.md`, `readme.md`,
    `particula/tests/condensation_latent_heat_docs_test.py`, and E4-F7 plan
    sections.
  - Tests: `pytest particula/tests/condensation_latent_heat_docs_test.py -q
    -Werror` (22 passed, 2026-07-15; no Warp/CUDA imports).
