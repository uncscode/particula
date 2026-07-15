# Feature Roadmap

This roadmap summarizes the current direction for Particula feature
development. It is a guide for users and contributors, not a fixed release
schedule.

## Current Focus

- **Strategy-based physics systems**: Continue standardizing dynamics modules
  around strategy, builder, and factory APIs.
- **Runnable workflows**: Make condensation, coagulation, wall loss, and related
  processes easier to compose in clear simulation pipelines.
- **Particle and gas data containers**: Complete migration from legacy facades
  toward explicit `ParticleData` and `GasData` containers.
- **GPU-backed particle-resolved simulations**: Move the existing Warp kernels
  from tested lower-level APIs toward documented, high-level simulation
  workflows.
- **Documentation and examples**: Expand practical examples that connect feature
  guides, theory pages, and runnable notebooks.

## Planned Improvements

### Dynamics Systems

- Add more high-level examples for combining condensation, coagulation, and wall
  loss in shared time-stepping loops.
- Improve guidance for choosing distribution types across discrete,
  continuous-PDF, and particle-resolved simulations.
- Continue exposing new physics models through consistent builders and
  factories.

### Data Model Migration

- Reduce reliance on legacy `ParticleRepresentation` and `GasSpecies` facade
  patterns where newer data containers provide clearer state management.
- Keep migration documentation up to date as APIs stabilize.
- Identify remaining compatibility layers that can be simplified before a
  stable major release.
- Track details in the
  [data-oriented design and GPU roadmap](data-oriented-gpu.md).

### GPU Acceleration

- Integrate existing Warp condensation and Brownian coagulation kernels into
  higher-level user workflows.
- Add documented examples for GPU-resident particle simulations that avoid
  repeated CPU/GPU transfers.
- Define parity, performance, and fallback expectations for CPU, Warp CPU, and
  CUDA execution.
- Track details in the
  [data-oriented design and GPU roadmap](data-oriented-gpu.md).

### Examples and Education

- Add more end-to-end chamber simulation examples.
- Expand notebook coverage for feature systems that currently have only API
  documentation.
- Improve links between examples, feature guides, theory pages, and API
  reference material.

## Epic Status

The data-oriented and GPU work is tracked as a single ordered epic sequence
with explicit exit bars in the
[Data-Oriented Design and GPU Roadmap](data-oriented-gpu.md#epic-sequence-and-status).
Each epic targets roughly 5-10 features of 5-15 phases each; when an epic
meets its exit bar, the next pending epic in the sequence becomes active.

### Shipped

- [Epic A: Data-Model and Numerical Foundations](data-oriented-gpu.md#epic-a-data-model-and-numerical-foundations)
  (ADW plan E2) — container schemas, `EnvironmentData`, CPU↔GPU transfer
  boundary, precision baseline, and stiffness recommendation. Shipped
  artifacts:
    - [Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
      — shipped guide for the current container, transfer-helper, and support
      boundary baseline
    - [Data Containers example](../../Examples/Data_Containers/index.md) —
      canonical runnable entrypoint linked from the roadmap handoff notes
    - [Mass Precision Recommendation Report](mass-precision-study.md) —
      canonical reference for downstream dtype/schema decisions
    - [Condensation stiffness study baseline](condensation-stiffness-study.md)
      — recorded timestep grid, fixed-four P2-finalized direct GPU contract,
      inclusive threshold semantics, and accepted `(n_boxes,)` environment inputs
- [Epic B: Non-Isothermal Condensation Public API (CPU)](data-oriented-gpu.md#epic-b-non-isothermal-condensation-public-api-cpu)
  (ADW plan E1) — public builder/factory access, validation, and
  documentation for latent-heat condensation on the CPU reference path.
- [Epic C: GPU Kernel Correctness and Low-Level API Hardening](data-oriented-gpu.md#epic-c-gpu-kernel-correctness-and-low-level-api-hardening)
  (ADW plan E3) — persistent coagulation RNG state, bounded mixed-scale
  selector hardening, measured one-thread-per-box limits, documented direct
  kernel entry points, device-aware test policy, and the completed Epic B
  latent-heat example and integration baseline. Shipped artifacts:
    - [Direct GPU kernels quick-start](../../Examples/gpu_direct_kernels_quick_start.py)
      — explicit transfer and low-level condensation/coagulation kernel path
    - [CPU latent-heat condensation example](../../Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb)
      — runnable bookkeeping reference for future GPU parity
    - [Measured coagulation decision record](data-oriented-gpu.md#measured-decision-record-for-the-current-one-thread-per-box-path)
      — accepted many-box baseline and large single-box caution band

### Next Up

- [Epic D: GPU Condensation Physics Parity](data-oriented-gpu.md#epic-d-gpu-condensation-physics-parity)
  — the next epic in the sequence; not yet scheduled as an ADW plan.

### Pending

Listed in execution order; each becomes active when the previous epic ships.

- [Epic E: GPU Coagulation Physics Coverage](data-oriented-gpu.md#epic-e-gpu-coagulation-physics-coverage)
- [Epic F: GPU Process Completeness](data-oriented-gpu.md#epic-f-gpu-process-completeness)
- [Epic G: Backend Selection and GPU-Resident Simulation](data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation)
- [Epic H: Graph Capture and Performance](data-oriented-gpu.md#epic-h-graph-capture-and-performance)
- [Epic I: Differentiability and Global Optimization](data-oriented-gpu.md#epic-i-differentiability-and-global-optimization)
  — implementation companion:
  [Warp autodiff limitations](warp-autodiff-limitations.md)

## Roadmap Artifacts

This folder can also hold supporting artifacts for planning and examples, such
as design notes, example outputs, prototype workflows, and milestone-specific
pages.

- [Data-Oriented Design and GPU Roadmap](data-oriented-gpu.md)

Reference anchors used by other documentation:

- [Current container schema inventory](data-oriented-gpu.md#current-container-schema-inventory)
- [Authoritative field ownership decisions](data-oriented-gpu.md#authoritative-field-ownership-decisions)
- [Canonical shape conventions for container workflows](data-oriented-gpu.md#canonical-shape-conventions-for-container-workflows)
- [Final downstream handoff map for sibling features](data-oriented-gpu.md#final-downstream-handoff-map-for-sibling-features)
- [CPU↔GPU restore boundary for ordered gas metadata](data-oriented-gpu.md#cpugpu-restore-boundary-for-ordered-gas-metadata)
- [Current CPU execution limits for multi-box-ready containers](data-oriented-gpu.md#current-cpu-execution-limits-for-multi-box-ready-containers)

## Contribution Opportunities

- Add a new physics strategy with builder, factory, tests, and documentation.
- Convert lower-level utilities into feature-level examples that show complete
  workflows.
- Improve theory pages with citations, assumptions, and model limitations.
- Report missing documentation or unclear APIs in GitHub Issues or Discussions.

## How to Propose Changes

For small fixes, open a pull request directly. For larger features, start with a
GitHub Discussion so maintainers and users can align on scope, API shape, and
testing expectations before implementation.
