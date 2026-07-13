# Architecture Design

## High-Level Design

```text
E4-F1 vapor pressure + E4-F2 activity/surface + E4-F3 four-substep core
        + E4-F4 latent heat + E4-F5 gas conservation
        + E4-F6 cross-device evidence
                              |
                              v
                 verify final API and test evidence
                              |
          +-------------------+-------------------+
          v                   v                   v
 canonical contract    runnable example    troubleshooting/commands
          +-------------------+-------------------+
                              v
          README/index/roadmap links + documentation guardrail tests
```

The canonical contract remains in `data-containers-and-gpu-foundations.md`. Other pages provide task-specific detail or concise links, avoiding divergent copies of the support matrix.

## Data / API / Workflow Changes

- **Data Model:** No runtime schema changes. CPU containers remain authoritative; Warp mirrors and caller-owned sidecars remain explicit. `EnvironmentData` owns temperature, pressure, and saturation ratio. Ordered gas species names remain CPU-owned metadata.
- **API Surface:** No new production API is planned. Documentation reflects the final direct `particula.gpu.kernels` condensation API from E4-F1 through E4-F6, including supported numeric configuration, stable-shape fp64 buffers, in-place mutation, returned transfer/energy diagnostics, and validation-before-mutation behavior.
- **Workflow Hooks:** The example explicitly performs `to_warp_*` conversion, low-level execution, synchronization/checkpoint decisions, and `from_warp_*` restore. No kernel or runnable may hide CPU-to-Warp transfer, CPU vapor-pressure refresh, or backend fallback.
- **Evidence Gate:** Support claims are generated from focused tests, not inferred from code shape. Warp CPU is mandatory; CUDA is optional and skip-clean.

## Security & Compliance

This feature adds no credentials, network access, or permissions. Robustness requirements are scientific and operational: validate shapes, species order, numeric configuration, physical values, devices, and scratch buffers before mutation; state units and diagnostic aggregation; and avoid claims beyond measured fp64 evidence. Commands must not imply that optional CUDA is universally available.
