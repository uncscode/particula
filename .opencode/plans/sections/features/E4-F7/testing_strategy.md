# Testing Strategy

Every phase ships with its documentation regression coverage. Existing coverage thresholds remain unchanged.

## Per-Phase Approach

- **P1:** Update `particula/tests/condensation_latent_heat_docs_test.py` and related docs assertions. Verify final low-level claims, direct `particula.gpu.kernels` imports, explicit transfers, exactly four substeps, fixed-shape fp64 state, unsupported high-level behavior, and preservation of the CPU notebook contract.
- **P2:** Extend `particula/gpu/tests/gpu_direct_kernels_example_test.py`. Run the example on Warp CPU; assert output and state checks, lazy imports, explicit conversion/restore, caller-owned buffer reuse, and clean no-Warp behavior. Optional CUDA coverage skips when unavailable.
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

Physics parity, inventory conservation, and energy bookkeeping use separate explicit tolerances. Multi-box CPU references must be independently evaluated per box rather than inferred from storage shape. Coverage remains at least 80%, and test files retain the `*_test.py` convention.
