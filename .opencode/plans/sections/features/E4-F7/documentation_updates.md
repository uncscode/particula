# Documentation Updates

- `docs/Features/data-containers-and-gpu-foundations.md` — canonical GPU condensation support matrix, transfer contract, diagnostics, unsupported modes, troubleshooting, and reproduction commands.
- `docs/Features/particle-data-migration.md` — final configuration/signature examples and migration failure guidance, linked to the canonical matrix.
- `docs/Examples/gpu_direct_kernels_quick_start.py` — runnable explicit-transfer condensation workflow with Warp CPU default, reusable buffers, gas coupling, energy output, and checkpoint restore.
- `docs/Examples/index.md` — publish and describe the canonical example and parity walkthrough.
- `readme.md` — concise GPU condensation discovery entry and canonical command/link; do not duplicate the full matrix.
- `docs/Features/Roadmap/data-oriented-gpu.md` — mark E4 condensation parity shipped only after E4-F1 through E4-F7 pass their exit gates.
- `docs/Features/Roadmap/condensation-stiffness-study.md` — link final production guidance without weakening the exactly-four-substep, fixed-shape, or fp64 decision record.
- `particula/tests/condensation_latent_heat_docs_test.py` — deliberately replace stale future-work assertions with precise low-level support and unsupported high-level guardrails; preserve paired CPU notebook checks.
- `.opencode/guides/testing_guide.md` or `AGENTS.md` — update only if focused commands or device policy become durable contributor guidance.
- `.opencode/plans/sections/features/E4-F7/*.md` — update statuses, final names, commands, and resolved questions as implementation lands.
