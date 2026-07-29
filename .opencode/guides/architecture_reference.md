# Architecture Reference

**Project:** particula  
**Last Updated:** 2026-07-28

This reference summarizes the particula package structure and key architectural
conventions migrated from the legacy guide set.

## Package Map

```text
particula/
├── activity/          # Activity coefficients and phase separation
├── dynamics/          # Coagulation, condensation, wall loss
├── equilibria/        # Partitioning calculations
├── execution/         # Dependency-neutral execution selection
├── gas/               # Gas phase, species, vapor pressure
├── particles/         # Particle distributions and representations
├── util/              # Constants, validation, chemistry utilities
└── integration_tests/ # Cross-module integration tests
```

## Architectural Patterns

- Keep physics calculations in focused modules with clear units and citations.
- Use strategy, builder, and factory patterns where the package already uses
  them, especially in wall loss, vapor pressure, and representation code.
- Keep tests co-located with modules in `tests/` directories.
- Export public APIs deliberately through package `__init__.py` files.
- Keep validation close to public function boundaries.

## Execution Selection and Condensation State

`particula.execution` is a package. Its supported selection surface remains
the existing exact ten-name public export list; do not promote adapter modules
or state carriers through it or through top-level `particula`.

`particula.execution.adapters.condensation` contains concrete-only P2
condensation state carriers and shipped P3/P4 selected CPU/Warp adapters. The
carriers retain caller-owned CPU or resident Warp resources by identity and
perform construction-time, read-only metadata and ownership checks. The
adapters select no profile themselves: after exact profile and sidecar
preflight, CPU calls the supplied isothermal runnable once and Warp calls the
direct kernel once. They do not transfer, restore, synchronize, fall back, or
recover failures. Omitted direct-kernel property arrays retain the native
step's documented step-local fallback allocation behavior; callers own reusable
native sidecars, resource lifetime, synchronization, concurrency, and any
post-launch mutation or rollback semantics. These concrete names remain absent
from the package selection surface and top-level `particula`.

`particula.execution.gpu_session` is a separate concrete-only, direct-import
boundary. Its P1 `ResidentSession` retains already-resident caller-owned Warp
particle, gas, and environment containers, immutable dimensions, Warp `Device`
metadata, a CPU gas-name tuple, and immutable lifecycle vocabulary by identity
after fixed-cost read-only type/schema/dtype/shape/device metadata validation.
P2 adds the direct-import-only `setup_resident_session` factory: CPU-only local
preflight precedes exactly one particle, gas, then environment conversion. It
retains the ordered CPU gas-name tuple and publishes only the fully validated,
`ACTIVE` resident session. Selected-device runtime availability remains the
upstream E7-F6 responsibility. This boundary has no export from
`particula.execution`, its adapters package, or top-level `particula`; it adds
no fallback, synchronization, restoration, sidecars, or lifecycle operations.
P4 adds direct-import-only `ResidentStepGuard` and identity-only
`ResidentStepToken` bookkeeping beside the immutable P1 carrier. One exact
active session/registry binding permits one open token; completed-step count and
simulated time advance only after matching completion. The guard neither
executes adapters nor transfers, restores, synchronizes, allocates, resizes, or
falls back. Future checkpoint, restore, finalize, close, fault, conversion, and
resize/rebind boundaries must call `assert_step_closed()` before their own work;
P5/P6 retain those operations and their policy. Direct low-level helpers remain
outside the guard's interception. Later phases own operational lifecycle
semantics; direct kernel and adapter boundaries retain physical validation. See
[ADR-004](architecture/decisions/ADR-004-concrete-gpu-resident-session-boundary.md).

`particula.execution.gpu_resources` is a separate direct-import-only,
Warp-dependent concrete boundary beside `gpu_session`. Each `GPUResourceRegistry`
accepts exactly one exact `ACTIVE` `ResidentSession`, pins its dimensions,
device, lifecycle, and all primary-array identities, and rejects any later drift
before acquisition. It builds complete reusable native condensation,
coagulation, wall-loss, and nucleation sidecars; validates their complete fixed
dtype/shape/device/contiguity metadata; checks allocation sizes; and rejects
sidecar-to-sidecar and primary-to-sidecar aliasing. Compatible repeats return the
same views, records, and Warp arrays by identity. Ownership means pinned role
identity and nonaliasing, not allocator-provenance inference. Its typed
manifests and views remain concrete-only and absent from package exports. The
boundary performs no execution or selection, transfer/synchronization/
restoration, lifecycle change, transport allocation, process
physics/configuration, or RNG initialization, advancement, or reset.
Condensation thermodynamic entries are derived scratch storage only, never
thermodynamic configuration.
Its narrow `validate_pinned_session()` seam first requires exact session
identity, then reuses the existing active lifecycle/signature/schema validation
without acquisition, allocation, payload inspection, or mutation.

## Wall Loss

Wall loss strategies live in `particula.dynamics.wall_loss` and are exported
through `particula.dynamics`.

Key concepts:

- `WallLossStrategy` is the abstract base class.
- `WallLoss` wraps a strategy, splits `time_step` across `sub_steps`, clamps
  concentrations nonnegative, and composes with other runnables via `|`.
- `ChargedWallLossStrategy` adds image-charge enhancement, optional electric
  field drift, and neutral fallback when particle charge and field are zero.
- Supported distribution types are `"discrete"`, `"continuous_pdf"`, and
  `"particle_resolved"`.

## Nucleation

The public CPU-only nucleation API is exported through `particula.dynamics`.
P4 provides immutable activation/kinetic potential-rate strategies,
source-selection metadata, builders, and a factory. P5 provides the one-box
`Nucleation` runnable and `NucleationCommitConfig`.

`Nucleation` preserves the legacy `Aerosol` and backing-data identities. It
uses equal, gas-coupled substeps and is atomic per attempted substep, not across
the complete call. P2 source-demand planning and P3 particle-source
transactions remain concrete-only in `nucleation.particle_source`.

The concrete `particula.gpu.kernels.nucleation` module implements the
package-exported direct `nucleation_step_gpu`: P1 preflight, P2 admission, P3
fixed-slot staging, P4 resampling-first/scaling-fallback, and fused P5
selected-slot/gas-transfer commit. Only the step is exported; configuration,
records, sidecars, and helpers remain concrete-only. Callers own conversion,
same-device fixed-shape sidecars, device placement, and synchronization; a
successful call returns the identical particle and gas containers. P3 retains
demand beyond free capacity and reports ascending free-slot prefixes with `-1`
tails. Public rejection before P4 primitive entry preserves particle and gas
state, while documented P2--P4 sidecars may already have been written; no
rollback is promised after E6-F6 primitive entry or P5 writer launch. The direct
step has no hidden transfer, CPU fallback, resize/compaction, GPU `Runnable`, or
E6-F9 integration.

## Scientific Utilities

- Physical constants belong in `particula.util.constants`.
- Public numerical validation uses `particula.util.validate_inputs`.
- Chemical helper code lives under `particula/util/chemical/`.

## Documentation

Architecture guides and ADRs are under `.opencode/guides/architecture/`. Update
them when module boundaries, exported APIs, or major design patterns change.
