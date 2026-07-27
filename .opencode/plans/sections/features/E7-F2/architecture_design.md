# Architecture Design

## High-Level Design

E7-F2 is an adapter layer between the dependency-neutral E7-F1 execution
context and existing condensation implementations. Selection validates declared
capability before loading or invoking a concrete backend. The CPU path remains
the independent reference. The GPU adapter imports Warp-only configuration and
kernel code lazily or from a GPU-scoped module according to E7-F6 policy.

```text
ExecutionRequest(process=condensation, backend, device, capabilities)
                              |
                    CapabilityMatrix.require
                              |
               unsupported --+--> CapabilityError
                              |
                    CondensationAdapter
                     /                 \
          CPU state + runnable      Warp resident state
          MassCondensation          particles / gas / environment
                     \                 /
                      explicit backend call
                              |
                    ExecutionResult
          state identity + mutation/transfer metadata
```

No adapter uploads, restores, synchronizes, or catches a runtime error to run a
different backend. The Warp adapter validates the E7 state wrapper and then
delegates to `condensation_step_gpu`, which owns detailed kernel validation and
its documented partial-failure boundary.

## Shipped P1 Metadata Boundary

P1 adds a closed, immutable semantic catalogue directly to
`particula/execution.py`, before any future adapter boundary. A frozen
`CondensationConfiguration` maps through an explicit lookup to exactly four
requirements: execution mode, latent-heat choice, activity mode, and surface
mode. The catalogue declares all 36 CPU combinations and only the eight
equal-step, ideal/kappa, static/composition-weighted Warp-profile combinations.

The Warp profile device identifier is opaque catalogue metadata, not a native
device claim. P1 queries only `CapabilityMatrix.supports()` or `.require()`;
it does not create an execution context, resolve a device, select an adapter,
import Warp, or mutate state. Consequently, runtime availability, fallback,
native-device normalization, state ownership, and execution remain future-phase
concerns.

## Data / API / Workflow Changes

- **Data model:** Add immutable condensation capability/configuration values and
  typed CPU/Warp execution-state views. Reference existing containers and
  sidecars by identity; do not copy or alter their schemas.
- **Supported mapping:** CPU supports the declared `MassCondensation` runnable.
  Warp supports shipped direct isothermal behavior and optional latent heat;
  kappa water activity and surface-tension modes are admitted only when exactly
  representable by `CondensationActivitySurfaceConfig`. Staggered and
  unsupported BAT requests are capability errors.
- **API surface:** Register a condensation process adapter through E7-F1's typed
  registry. Expose only stable selection/configuration names allowed by E7-F6;
  retain concrete scratch and thermodynamic types at current module locations.
- **Inputs:** Time is finite and nonnegative. Warp state contains same-device
  particle/gas data and either an environment or valid direct temperature and
  pressure, plus thermodynamics and optional activity/scratch/thermal sidecars.
- **Returns/mutation:** CPU returns the runnable's `Aerosol`. Warp returns the
  identical particle state plus the whole-call finalized transfer through an
  `ExecutionResult`; particle masses, gas concentration, vapor pressure, and
  documented outputs mutate in place. `energy_transfer` remains caller-owned
  output and is not invented as a kernel tuple item.
- **Workflow hooks:** E7-F4 may later place these resources in resident session
  state; E7-F5 schedules the adapter after environment changes and before
  downstream consumers. Neither concern is implemented here.

## Validation and Failure Semantics

Selection-level validation runs before adapter invocation: request/process,
backend/device availability, capability/configuration, execution-state kind,
and finite time. Concrete Warp schema/value validation remains authoritative in
the shipped kernel and must retain its order. Pre-launch rejection is atomic.
Once a Warp substep commits, later proposal failure does not roll back earlier
work; the result and documentation must not promise transactionality.

## Security & Compliance

No credentials or network permissions are introduced. Adapters are registered
from typed trusted code, never dynamically imported from untrusted strings.
Fail-closed capability validation, bounded exports, explicit device ownership,
and prohibition of hidden transfers reduce resource and state-integrity risk.
