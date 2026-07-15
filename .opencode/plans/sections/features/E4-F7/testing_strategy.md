# Testing Strategy

Every phase ships with its documentation regression coverage. Existing coverage thresholds remain unchanged.

## Per-Phase Approach

- **P1 (completed, Issue #1314):** `particula/tests/condensation_latent_heat_docs_test.py` now provides text-only assertions for the canonical foundations configuration, schemas/input variants, lifecycle/ownership semantics, validation and shipped boundaries, and migration guidance. It retains the CPU notebook/publication checks and requires no Warp or CUDA device.
- **P2 (completed, Issue #1315):** `particula/gpu/tests/gpu_direct_kernels_example_test.py` covers exact deterministic output, forced no-Warp import isolation, the lazy public-step/concrete-sidecar loader, mocked explicit conversion/restore and same-object sidecar reuse, and failure propagation without restore. Its guarded real Warp-CPU test checks restored shapes/names, two-call sidecar identity, nonzero finalized transfer/energy, the unweighted energy identity, and concentration-weighted particle/gas conservation. CUDA is not required.
- **P3 (completed, Issue #1316):** `particula/tests/condensation_latent_heat_docs_test.py` provides scoped text-only checks for the foundations troubleshooting and command H3, migration pointer, exact command targets/flags, README anchor discovery, prohibited claims, and distinct parity/inventory/energy evidence. The published baseline is Warp `device="cpu"`; CUDA is optional/local and its marker command skips cleanly when unavailable.
- **P4:** Validate markdown links, roadmap language, example-index discovery, and the complete focused documentation suite.

## Published Reproduction Baseline

```bash
python docs/Examples/gpu_direct_kernels_quick_start.py
pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q -Werror
pytest particula/integration_tests/condensation_latent_heat_conservation_test.py -q
pytest particula/integration_tests/condensation_particle_resolved_test.py -q
pytest particula/tests/condensation_latent_heat_docs_test.py -q -Werror
```

The optional/local CUDA command is `pytest particula/gpu/kernels/tests/condensation_test.py -q -m "warp and cuda" -Werror`; it is additive to the required Warp `device="cpu"` baseline and skips cleanly when CUDA is unavailable. Primary direct CPU-oracle particle-mass/gas-concentration parity, particle-plus-gas inventory conservation, and latent-heat energy/bookkeeping remain distinct evidence classes; none proves either of the others.

Issue #1314 validation command: `pytest particula/tests/condensation_latent_heat_docs_test.py -q -Werror`. Issue #1315 validation command: `pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q -Werror`.

Physics parity, inventory conservation, and energy bookkeeping use separate explicit tolerances. Multi-box CPU references must be independently evaluated per box rather than inferred from storage shape. Coverage remains at least 80%, and test files retain the `*_test.py` convention.
