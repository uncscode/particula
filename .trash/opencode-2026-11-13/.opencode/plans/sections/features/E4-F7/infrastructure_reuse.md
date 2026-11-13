# Infrastructure Reuse

- `docs/Features/data-containers-and-gpu-foundations.md:164-187,237-295,327-366` — canonical low-level API, transfer boundary, support matrix, and troubleshooting surface; extend rather than duplicate.
- `docs/Features/particle-data-migration.md:367-416,476-491` — detailed environment input and validation guidance; update final signatures and link back to the canonical contract.
- `docs/Examples/gpu_direct_kernels_quick_start.py:79-199` — existing lazy-import, Warp CPU default, explicit conversion, kernel call, and restore pattern.
- `particula/gpu/tests/gpu_direct_kernels_example_test.py:18-188` — exact-output, behavior, and no-Warp regression pattern for the example.
- `particula/gpu/kernels/condensation.py:387-570` — implementation and docstring source for final argument, mutation, buffer, and return semantics after E4-F1 through E4-F6 land.
- `particula/gpu/conversion.py:198-375` — explicit particle, gas, and environment conversion boundary; preserve CPU-owned gas names and dropped GPU-only helper state.
- `particula/gpu/kernels/tests/condensation_test.py` — canonical focused GPU contract and Warp CPU reproduction target.
- `particula/integration_tests/condensation_latent_heat_conservation_test.py:231-299` and `condensation_particle_resolved_test.py:74-115` — independent CPU references for energy and activity/surface behavior.
- `particula/tests/condensation_latent_heat_docs_test.py:8-144` — issue 1272 wording and paired-notebook guardrails; revise assertions deliberately instead of deleting them.
- `docs/Features/Roadmap/condensation-stiffness-study.md:119-204` — authoritative exactly-four-substep, stable-shape, and fp64 rationale.
- `.opencode/guides/testing_guide.md:166-228` — Warp CPU baseline, optional CUDA skip policy, and distinct parity/conservation tolerances.

Follow the repository pattern of one canonical detailed contract with concise links from `readme.md`, migration material, examples index, and roadmap pages.
