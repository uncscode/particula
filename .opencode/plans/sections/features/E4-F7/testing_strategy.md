# Testing Strategy

Every phase ships with its documentation regression coverage. Existing coverage thresholds remain unchanged.

## Per-Phase Approach

- **P1 (completed, Issue #1314):** `particula/tests/condensation_latent_heat_docs_test.py` now provides text-only assertions for the canonical foundations configuration, schemas/input variants, lifecycle/ownership semantics, validation and shipped boundaries, and migration guidance. It retains the CPU notebook/publication checks and requires no Warp or CUDA device.
- **P2 (completed, Issue #1315):** `particula/gpu/tests/gpu_direct_kernels_example_test.py` covers exact deterministic output, forced no-Warp import isolation, the lazy public-step/concrete-sidecar loader, mocked explicit conversion/restore and same-object sidecar reuse, and failure propagation without restore. Its guarded real Warp-CPU test checks restored shapes/names, two-call sidecar identity, nonzero finalized transfer/energy, the unweighted energy identity, and concentration-weighted particle/gas conservation. CUDA is not required.
- **P3:** Run command/path checks and focused behavior suites: `condensation_test.py`, final stiffness tests, CPU latent-heat and particle-resolved references, and documentation tests. Run the primary GPU suite with `-Werror`.
- **P4:** Validate markdown links, roadmap language, example-index discovery, and the complete focused documentation suite.

## Published Reproduction Baseline

```bash
python docs/Examples/gpu_direct_kernels_quick_start.py
pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q
pytest particula/gpu/kernels/tests/condensation_test.py -q
pytest particula/gpu/kernels/tests/condensation_stiffness_test.py -q
pytest particula/integration_tests/condensation_latent_heat_conservation_test.py -q
pytest particula/integration_tests/condensation_particle_resolved_test.py -q
pytest particula/tests/condensation_latent_heat_docs_test.py -q
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
```

Before publication, reconcile command names with the files and markers delivered by E4-F1 through E4-F6. Warp CPU is the required baseline. Publish a CUDA marker command only if E4-F6 defines it; label it optional/local and require a clean skip without CUDA.

Issue #1314 validation command: `pytest particula/tests/condensation_latent_heat_docs_test.py -q -Werror`. Issue #1315 validation command: `pytest particula/gpu/tests/gpu_direct_kernels_example_test.py -q -Werror`.

Physics parity, inventory conservation, and energy bookkeeping use separate explicit tolerances. Multi-box CPU references must be independently evaluated per box rather than inferred from storage shape. Coverage remains at least 80%, and test files retain the `*_test.py` convention.
