# Architecture Design

## High-Level Design

```text
E4-F1 vapor pressure + E4-F2 activity/surface + E4-F3 four-substep core
        + E4-F4 latent heat + E4-F5 gas conservation
        + E4-F6 cross-device evidence
                              |
                              v
                  publish bounded contract and runnable example
                              |
          +-------------------+-------------------+
           v                   v                   v
   canonical contract    runnable example    troubleshooting/commands
          +-------------------+-------------------+
                              v
       migration/README discovery + focused text-only regression tests
```

The canonical contract is published in `docs/Features/data-containers-and-gpu-foundations.md`, with concise migration guidance in `docs/Features/particle-data-migration.md`. The runnable companion is `docs/Examples/gpu_direct_kernels_quick_start.py`: it keeps CPU fixtures and conversion/restore boundaries explicit, lazily imports the public step plus concrete sidecar types only in the Warp branch, and reuses caller-owned fp64 buffers across two direct calls. `particula/gpu/tests/gpu_direct_kernels_example_test.py` protects the no-Warp, mocked-contract, failure, and guarded real Warp-CPU paths.

## Data / API / Workflow Changes

- **Data Model:** No runtime schema changes. CPU containers remain authoritative; Warp mirrors and caller-owned sidecars remain explicit. `EnvironmentData` owns temperature, pressure, and saturation ratio. Ordered gas species names remain CPU-owned metadata.
- **API Surface:** Issues #1314 and #1315 made no production API change. The documentation and example reflect the direct `particula.gpu.kernels` condensation API: required `thermodynamics=`, a two-item `(particles, mass_transfer)` return, in-place particle/gas mutation, caller-owned diagnostics, stable-shape fp64 buffers, and validation-before-mutation behavior.
- **Workflow Hooks:** The example explicitly performs `to_warp_*` conversion, two low-level calls with identical sidecars, and `from_warp_*` final-checkpoint restores. No kernel or runnable may hide CPU-to-Warp transfer, CPU vapor-pressure refresh, or backend fallback.
- **Evidence Gate:** Support claims are generated from focused tests, not inferred from code shape. Issue #1316 added the foundations command matrix, migration pointer, README discovery link, and scoped text-only checks. Warp `device="cpu"` is the required baseline; the CUDA-marker command is optional/local and skip-clean when CUDA is unavailable.

## Security & Compliance

This feature adds no credentials, network access, or permissions. Robustness requirements are scientific and operational: validate shapes, species order, numeric configuration, physical values, devices, and scratch buffers before mutation; state units and diagnostic aggregation; and avoid claims beyond measured fp64 evidence. Commands must not imply that optional CUDA is universally available.
