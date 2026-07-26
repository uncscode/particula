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
- **GPU process completeness**: Maintain the shipped GPU dilution, wall loss,
  nucleation, and fixed-capacity slot-management contracts from
  [Epic F](data-oriented-gpu.md#epic-f-gpu-process-completeness).
- **GPU-resident user workflows**: Advance backend selection and high-level
  simulation integration through active
  [Epic G](data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation).
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
      — explicit-transfer, two-call low-level condensation path with reused
      caller-owned fp64 sidecars; it does not invoke coagulation or configure
      RNG state
    - [CPU latent-heat condensation example](../../Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb)
      — runnable bookkeeping reference for future GPU parity
    - [Measured coagulation decision record](data-oriented-gpu.md#measured-decision-record-for-the-current-one-thread-per-box-path)
      — accepted many-box baseline and large single-box caution band
- [Epic D: GPU Condensation Physics Parity](data-oriented-gpu.md#epic-d-gpu-condensation-physics-parity)
  (ADW plan E4) is the shipped bounded low-level direct-condensation
  publication. It includes constant and Buck vapor-pressure refresh, ideal and
  kappa activity, static and composition-weighted surface tension, fixed-four
  P2 inventory finalization, gas coupling and conservation, latent-heat rate
  correction and signed energy diagnostics, reusable caller-owned fp64
   sidecars, device-aware parity evidence, and the published direct-kernel
   support contract and example. It does not provide high-level runnable
   integration or general CPU-strategy parity.
     - Independent condensation walkthrough
       (`docs/Examples/gpu_condensation_parity_walkthrough.py`)
       — fixed-four-substep low-level direct-kernel physics, conservation, and
         energy evidence, with Warp CPU as the installed-Warp baseline; CUDA is
         optional additive evidence
      - Downstream condensation ownership record
        (`condensation-parity-walkthrough.md`)
       — deferred work only; it does not alter Epic D production capability or
         activate later epics
     - The walkthrough's caller-owned, write-only `energy_transfer` diagnostic
       is not a return value or temperature feedback (`kg * J/kg = J`). It
       preserves fixed-four-substep direct-kernel scope: no strategy/`Runnable`
       parity, adaptive stepping, graph capture/replay, broad autodiff, or
       performance claim.
     - Focused evidence commands:
       `python docs/Examples/gpu_condensation_parity_walkthrough.py`,
       `pytest particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py -q -Werror`,
      and `pytest particula/tests/condensation_parity_walkthrough_docs_test.py -q -Werror`.
- [Epic E: GPU Coagulation Physics Coverage](data-oriented-gpu.md#epic-e-gpu-coagulation-physics-coverage)
  (ADW plan E5) and all nine feature plans E5-F1 through E5-F9 shipped on
  2026-07-20. The bounded direct-kernel release covers Brownian, charged
  hard-sphere, SP2016 sedimentation, and ST1956 turbulent-shear terms; approved
  additive masks; charge-conserving merges; deterministic and stochastic
  validation; explicit-transfer documentation and examples; and the ordered
  closeout gate. High-level `Runnable` integration remains deferred.
- [Epic F: GPU Process Completeness](data-oriented-gpu.md#epic-f-gpu-process-completeness)
  (ADW plan E6) and all nine feature plans E6-F1 through E6-F9 are shipped.
  The release supplies the bounded direct dilution, wall-loss, fixed-slot,
  exhaustion, and nucleation contracts plus explicit five-call sequence
  evidence. High-level GPU orchestration remains deferred to Epic G.

### E6 roadmap inventory

| ID | Title | Status text |
| --- | --- | --- |
| [`E6`](../../../.opencode/plans/epics/E6.json) | GPU Process Completeness | Shipped |
| [`E6-F1`](../../../.opencode/plans/features/E6-F1.json) | CPU Dilution Strategy and Runnable Reference | Shipped |
| [`E6-F2`](../../../.opencode/plans/features/E6-F2.json) | Direct GPU Dilution with CPU Parity | Shipped |
| [`E6-F3`](../../../.opencode/plans/features/E6-F3.json) | Neutral Spherical and Rectangular GPU Wall Loss | Shipped |
| [`E6-F4`](../../../.opencode/plans/features/E6-F4.json) | Charged GPU Wall Loss with Neutral Fallback | Shipped |
| [`E6-F5`](../../../.opencode/plans/features/E6-F5.json) | CPU and GPU Fixed-Slot Activation and Diagnostics | Shipped |
| [`E6-F6`](../../../.opencode/plans/features/E6-F6.json) | CPU and GPU Slot Exhaustion Policies | Shipped |
| [`E6-F7`](../../../.opencode/plans/features/E6-F7.json) | CPU Nucleation and Particle-Source Process | Shipped |
| [`E6-F8`](../../../.opencode/plans/features/E6-F8.json) | Direct GPU Nucleation Process | Shipped |
| [`E6-F9`](../../../.opencode/plans/features/E6-F9.json) | Integrated Validation Documentation and Epic Closeout | Shipped |

The private [P2 sequence evidence](../../../particula/gpu/tests/process_sequence_test.py),
[P3 source](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_complete_process_sequence.py),
and [P3 regression](../../../particula/gpu/tests/gpu_complete_process_sequence_example_test.py)
document an explicit five-call sequence. Its `condensation_step_gpu`,
`coagulation_step_gpu`, `dilution_step_gpu`, `wall_loss_step_gpu`, and
`nucleation_step_gpu` order is illustrative, not a production coordinator.

### E5 roadmap inventory

| ID | Title | Status text |
| --- | --- | --- |
| `E5` | GPU Coagulation Physics Coverage | Shipped |
| `E5-F1` | Mechanism Configuration and Sampling Contract | Shipped |
| `E5-F2` | Charged Pair Physics and Charge-Conserving Merges | Shipped |
| `E5-F3` | Charged and Brownian-Plus-Charged GPU Execution | Shipped |
| `E5-F4` | SP2016 Sedimentation GPU Execution | Shipped |
| `E5-F5` | ST1956 Turbulent-Shear GPU Execution | Shipped |
| `E5-F6` | Single-Pass Additive Multi-Mechanism Coagulation | Shipped |
| `E5-F7` | Cross-Mechanism GPU Validation Matrix | Shipped |
| `E5-F8` | Independent CPU-Warp Condensation Walkthrough | Shipped |
| `E5-F9` | GPU Coagulation Support Documentation and Epic Closeout | Shipped |

All E5 features are shipped. The public E5 cross-links are:

- E5-F7's [GPU coagulation validation record](coagulation-validation.md).
- E5-F8's
  [condensation parity walkthrough ownership record](condensation-parity-walkthrough.md)
  and runnable
  [GPU condensation parity walkthrough](../../Examples/gpu_condensation_parity_walkthrough.py).
- E5-F9's
  [canonical GPU coagulation ownership contract](../data-containers-and-gpu-foundations.md#gpu-coagulation-configuration-and-sidecar-ownership),
  [direct-kernel support guide](../coagulation_strategy_system.md#gpu-direct-kernel-foundations-and-limitations),
  and
  [direct GPU coagulation example](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_coagulation_direct.py).

E5, E6, and all of their feature plans are shipped. Epic G is active next.

### Active

- [Epic G: Backend Selection and GPU-Resident Simulation](data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation)
  follows Epic F. It will expose backend selection through user-facing APIs and
  keep complete simulation loops resident on the selected device between
  checkpoints.

### Pending

Listed in execution order; each becomes active when the previous epic ships.

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
