# Feature Roadmap

This roadmap summarizes the current direction for Particula feature
development. It is a guide for users and contributors, not a fixed release
schedule.

## Release and migration sequence

Maintainer decision, 2026-09-27: graph capture and capture capability through graph-capture documentation are Shipped.
The historical release sequence was:

1. Merge the graph capture closeout and roadmap update.
2. Release **0.2.13**, retaining both the legacy `ParticleRepresentation` /
   `GasSpecies` facades and the `ParticleData` / `GasData` / `EnvironmentData`
   container workflow.
3. Deliver **0.3.0** as the breaking container-first migration: remove the
   legacy facades, migrate dependent workflows, and update all affected
   documentation, examples, imports, and tests.
4. Continue Epic I, differentiability and initial-state optimization, after
   the migration milestone.

The current v0.3.0 development tree has removed the facades and temporary
untagged execution paths. Final validation and review are tracked in the
[native migration closeout](data-native-v030-closeout.md). This roadmap does
not publish a release. The [closeout record](graph-capture-closeout.md) records the maintainer's
plan closure separately from unavailable CUDA measurement evidence.

## Current Focus

- **Strategy-based physics systems**: Continue standardizing dynamics modules
  around strategy, builder, and factory APIs.
- **Runnable workflows**: Make condensation, coagulation, wall loss, and related
  processes easier to compose in clear simulation pipelines.
- **Particle and gas data containers**: Validate the native-only
  `ParticleData`, `GasData` and `EnvironmentData` workflow after facade removal.
- **GPU process completeness**: Maintain the shipped GPU dilution, wall loss,
  nucleation, and fixed-capacity slot-management contracts from
  [Epic F](data-oriented-gpu.md#epic-f-gpu-process-completeness).
- **GPU-resident user workflows**: Maintain the shipped backend-selection,
  resident scheduling, communication, checkpoint/restart, diagnostics, and RNG
  contracts from
  [Epic G](data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation).
- **Graph capture and performance**: Maintain shipped
  [Epic H](data-oriented-gpu.md#epic-h-graph-capture-and-performance)'s bounded
  capture/replay and documentation tracks. Use the [native-CUDA
  example](https://github.com/uncscode/particula/blob/main/docs/Examples/gpu_resident_graph_capture.py), [operator
  runbook](../gpu_graph_capture.md), and [closeout
  record](graph-capture-closeout.md). CUDA profiling owns profiling and
  machine-bounded recommendations; graph-capture documentation owns the example, runbook,
  limitations, and closeout. The closeout remains the sole evidence authority;
  measured performance results remain unavailable and unshipped until reviewed.
  No CPU or Warp-CPU fallback is provided.
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

- Keep supported construction and execution on explicit native containers.
- Keep migration documentation and removed-boundary tests aligned with the
  native API; historical facade names are migration references only.
- Complete the v0.3.0 scientific, installed-package and example validation before
  final review and release approval.
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
  — container schemas, `EnvironmentData`, CPU↔GPU transfer
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
      — recorded timestep grid, fixed-four inventory-limited direct GPU contract,
      inclusive threshold semantics, and accepted `(n_boxes,)` environment inputs
- [Epic B: Non-Isothermal Condensation Public API (CPU)](data-oriented-gpu.md#epic-b-non-isothermal-condensation-public-api-cpu)
  — public builder/factory access, validation, and
  documentation for latent-heat condensation on the CPU reference path.
- [Epic C: GPU Kernel Correctness and Low-Level API Hardening](data-oriented-gpu.md#epic-c-gpu-kernel-correctness-and-low-level-api-hardening)
  — persistent coagulation RNG state, bounded mixed-scale
  selector hardening, measured one-thread-per-box limits, documented direct
  kernel entry points, device-aware test policy, and the completed Epic B
  latent-heat example and integration baseline. Shipped artifacts:
    - [Direct GPU kernels quick-start](https://github.com/uncscode/particula/blob/main/docs/Examples/gpu_direct_kernels_quick_start.py)
      — explicit-transfer, two-call low-level condensation path with reused
      caller-owned fp64 sidecars; it does not invoke coagulation or configure
      RNG state
    - [CPU latent-heat condensation example](../../Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb)
      — runnable bookkeeping reference for future GPU parity
    - [Measured coagulation decision record](data-oriented-gpu.md#measured-decision-record-for-the-current-one-thread-per-box-path)
      — accepted many-box baseline and large single-box caution band
- [Epic D: GPU Condensation Physics Parity](data-oriented-gpu.md#epic-d-gpu-condensation-physics-parity)
  is the shipped bounded low-level direct-condensation
  publication. It includes constant and Buck vapor-pressure refresh, ideal and
  kappa activity, static and composition-weighted surface tension, fixed-four
  numerical validation inventory finalization, gas coupling and conservation, latent-heat rate
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
       `python docs/Examples/gpu_condensation_parity_walkthrough.py` and
       `pytest examples_tests/gpu_condensation_parity_walkthrough_test.py -q -Werror`.
- [Epic E: GPU Coagulation Physics Coverage](data-oriented-gpu.md#epic-e-gpu-coagulation-physics-coverage)
  and all nine feature plans coagulation sampling through coagulation support documentation shipped on
  2026-07-20. The bounded direct-kernel release covers Brownian, charged
  hard-sphere, SP2016 sedimentation, and ST1956 turbulent-shear terms; approved
  additive masks; charge-conserving merges; deterministic and stochastic
  validation; explicit-transfer documentation and examples; and the ordered
  closeout gate. High-level `Runnable` integration remains deferred.
- [Epic F: GPU Process Completeness](data-oriented-gpu.md#epic-f-gpu-process-completeness)
  and all nine feature plans CPU dilution through direct-process integration are shipped.
  The release supplies the bounded direct dilution, wall-loss, fixed-slot,
  exhaustion, and nucleation contracts plus explicit five-call sequence
  evidence. High-level GPU orchestration remains deferred to Epic G.
- [Epic G: Backend Selection and GPU-Resident Simulation](data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation)
  and all nine feature plans backend selection through resident diagnostics shipped on
  2026-08-30. The release closes backend selection, explicit fallback policy,
  bounded process adapters, resident sessions and checkpoints, deterministic
  scheduling, prescribed communication and volume evolution, persistent RNG
  restart semantics, diagnostics, full-loop regressions, and documentation.

### GPU process completeness roadmap inventory

| ID | Title | Status text |
| --- | --- | --- |
| `GPU process completeness` | GPU Process Completeness | Shipped |
| `CPU dilution` | CPU Dilution Strategy and Runnable Reference | Shipped |
| `GPU dilution` | Direct GPU Dilution with CPU Parity | Shipped |
| `neutral GPU wall loss` | Neutral Spherical and Rectangular GPU Wall Loss | Shipped |
| `charged GPU wall loss` | Charged GPU Wall Loss with Neutral Fallback | Shipped |
| `fixed-slot management` | CPU and GPU Fixed-Slot Activation and Diagnostics | Shipped |
| `slot exhaustion` | CPU and GPU Slot Exhaustion Policies | Shipped |
| `CPU nucleation` | CPU Nucleation and Particle-Source Process | Shipped |
| `direct GPU nucleation` | Direct GPU Nucleation Process | Shipped |
| `direct-process integration` | Integrated Validation Documentation and Epic Closeout | Shipped |

The private [numerical validation sequence evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py),
[integration validation source](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_complete_process_sequence.py),
and [example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py)
document an explicit five-call sequence. Its `condensation_step_gpu`,
`coagulation_step_gpu`, `dilution_step_gpu`, `wall_loss_step_gpu`, and
`nucleation_step_gpu` order is illustrative, not a production coordinator.

### GPU coagulation roadmap inventory

| ID | Title | Status text |
| --- | --- | --- |
| `GPU coagulation` | GPU Coagulation Physics Coverage | Shipped |
| `coagulation sampling` | Mechanism Configuration and Sampling Contract | Shipped |
| `charged pair rates` | Charged Pair Physics and Charge-Conserving Merges | Shipped |
| `charged coagulation` | Charged and Brownian-Plus-Charged GPU Execution | Shipped |
| `sedimentation coagulation` | SP2016 Sedimentation GPU Execution | Shipped |
| `turbulent-shear coagulation` | ST1956 Turbulent-Shear GPU Execution | Shipped |
| `additive coagulation mechanisms` | Single-Pass Additive Multi-Mechanism Coagulation | Shipped |
| `coagulation validation` | Cross-Mechanism GPU Validation Matrix | Shipped |
| `independent condensation walkthrough` | Independent CPU-Warp Condensation Walkthrough | Shipped |
| `coagulation support documentation` | GPU Coagulation Support Documentation and Epic Closeout | Shipped |

All GPU coagulation features are shipped. The public GPU coagulation cross-links are:

- Coagulation validation's [GPU coagulation validation record](coagulation-validation.md).
- Independent condensation walkthrough's
  [condensation parity walkthrough ownership record](condensation-parity-walkthrough.md)
  and runnable
  [GPU condensation parity walkthrough](https://github.com/uncscode/particula/blob/main/docs/Examples/gpu_condensation_parity_walkthrough.py).
- Coagulation support documentation's
  [canonical GPU coagulation ownership contract](../data-containers-and-gpu-foundations.md#gpu-coagulation-configuration-and-sidecar-ownership),
  [direct-kernel support guide](../coagulation_strategy_system.md#gpu-direct-kernel-foundations-and-limitations),
  and
  [direct GPU coagulation example](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_coagulation_direct.py).

### Resident execution roadmap inventory

| ID | Title | Status text |
| --- | --- | --- |
| `resident execution` | Backend Selection and GPU-Resident Simulation | Shipped |
| `backend selection` | Backend-Selection and Execution-Context API | Shipped |
| `selected condensation` | Backend-Selected Condensation | Shipped |
| `selected Brownian coagulation` | Backend-Selected Brownian Coagulation | Shipped |
| `resident sessions` | GPU-Resident Session State and Checkpoints | Shipped |
| `resident scheduling` | Deterministic Full-Process Scheduling | Shipped |
| `execution capability policy` | Fallback, Capability Errors, and API Stability | Shipped |
| `communication and volume evolution` | Multi-Box Communication and Volume Evolution | Shipped |
| `resident RNG and restart` | Persistent Per-Box RNG Streams and Restart Semantics | Shipped |
| `resident diagnostics` | Diagnostics, Full-Loop Regressions, Documentation, and Closeout | Shipped |

GPU coagulation, GPU process completeness, resident execution, graph capture, and all of their feature plans are shipped.

### Graph capture roadmap inventory

| ID | Title | Status text |
| --- | --- | --- |
| `graph capture` | Graph Capture and Performance | Shipped |
| `capture capability` | Graph-Capture Capability and Lifecycle Contracts | Shipped |
| `capture-ready enqueue` | Capture-Ready Device Enqueue Paths | Shipped |
| `resource preallocation` | Registry Preallocation, Identity Reuse, and Byte Accounting | Shipped |
| `capture and replay` | Resident Graph Capture and Guarded Replay Lifecycle | Shipped |
| `captured-loop validation` | Captured Full-Loop Parity and Lifecycle Validation | Shipped |
| `resident benchmark study` | Multi-Box Scaling Benchmarks and Memory-Budget Evidence | Shipped |
| `CUDA profiling` | CUDA Profiling and Machine-Bounded Performance Decisions | Shipped |
| `graph-capture documentation` | Graph-Capture Example, Runbook, Limitations, and Closeout | Shipped |

- [Epic H: Graph Capture and Performance](data-oriented-gpu.md#epic-h-graph-capture-and-performance)
  and its eight features were closed as Shipped by maintainer decision on
  2026-09-27. Reviewed CUDA measurements remain unavailable; plan closure
  does not establish a measured performance claim. Graph-capture documentation owns the example,
  runbook, limitations, and closeout, with no
  CPU or Warp-CPU fallback. The [native-CUDA example](https://github.com/uncscode/particula/blob/main/docs/Examples/gpu_resident_graph_capture.py),
  [operator runbook](../gpu_graph_capture.md), and [integration validation closeout record](graph-capture-closeout.md)
  are the discoverable references; the closeout alone owns integration validation status/evidence.

### Pending

The 0.2.13 release and 0.3.0 migration take precedence over the next epic.

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
