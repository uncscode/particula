# Documentation Updates

- **Shipped (P1, issue #1372):** Updated
  `docs/Features/data-containers-and-gpu-foundations.md` as the canonical
  direct GPU coagulation ownership and preflight contract. It records supported
  particle-resolved masks, caller-owned state, explicit transfer boundaries,
  installed-Warp CPU baseline, and optional guarded CUDA evidence.
- **Shipped (P1, issue #1372):** Replaced the GPU subsection in
  `docs/Features/coagulation_strategy_system.md` with the bounded direct-kernel
  mechanism/mask matrix, exclusions, and focused validation commands.
- **Shipped (P1, issue #1372):** Added the stdlib-only regression coverage in
  `particula/tests/gpu_coagulation_docs_test.py` and extended the exact command
  matrix assertions in `particula/tests/condensation_latent_heat_docs_test.py`.
- **Shipped (P2, issue #1373):** Added
  `docs/Examples/gpu_coagulation_direct.py`, a Warp-CPU-default direct
  Brownian example with explicit CPU↔Warp transfer boundaries, caller-owned
  collision/persistent-RNG sidecars, exactly two calls, lazy no-Warp behavior,
  and no fallback. `docs/index.md` now links the example and states its bounded
  support claims.
- **Documented P3 work (issue #1374; phase remains Not Started):** Reconciled
  `docs/Features/Roadmap/data-oriented-gpu.md` and
  `docs/Features/Roadmap/index.md` with one matching E5 inventory containing
  E5 and E5-F1 through E5-F9, their canonical statuses, and active/pre-closeout
  wording. E5 and E5-F9 remain active; P4 remains the closeout phase and Epic F
  remains pending.
- **Documented P3 work (issue #1374; phase remains Not Started):** Linked E5-F7's canonical validation artifact at
  `docs/Features/Roadmap/coagulation-validation.md` and E5-F8's
  `docs/Features/Roadmap/condensation-parity-walkthrough.md` plus
  `docs/Examples/gpu_condensation_parity_walkthrough.py` without duplicating
  their content, and added hardware-free regression coverage in
  `particula/tests/gpu_coagulation_docs_test.py` for record equality,
  uniqueness, status wording, and local-link resolution.
- Update E5/E5-F9 plan phase/status records only after the closeout gate passes.
