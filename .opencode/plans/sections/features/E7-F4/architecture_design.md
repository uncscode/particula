# Architecture Design

## High-Level Design

The resident session is a resource/lifecycle object in the E7-F1 execution
layer. It owns references to mutable Warp state and process resources while
keeping configuration and CPU-only metadata explicit. It is not a process
scheduler and never invokes physics by itself.

```text
CPU ParticleData + GasData + EnvironmentData + ordered metadata
                           |
          E7-F1 request + E7-F6 capability/availability validation
                           |
                           v
                 ResidentSession.setup(device)
                    |      |       |
                    |      |       +-- allocate/validate SidecarRegistry
                    |      +---------- one to_warp_* call per container
                    +----------------- freeze B/N/S/device metadata
                           |
                 ACTIVE resident state
                           |
             E7-F5 scheduler-facing lifecycle hooks
             (no restore, no bulk transfer, no implicit sync)
                           |
              +------------+-------------+
              |                          |
       checkpoint()                 finalize()
       sync once                    sync once
       restore x3                   restore x3
       ACTIVE remains               FINALIZED terminal
              |
       Checkpoint(state + metadata + opaque mutable resources)
              |
       explicit restart -> new setup/upload boundary
```

## Data / API / Workflow Changes

- **Data model:** Add a typed `ResidentSession`, immutable
  `ResidentDimensions`, lifecycle enum, `SidecarRegistry`, and immutable
  checkpoint record. The session references existing `WarpParticleData`,
  `WarpGasData`, and `WarpEnvironmentData`; no container schema changes are
  required.
- **Metadata:** Retain ordered gas names, backend/device identity, shape tuple,
  simulation step/time counters, schema version, and process-resource manifest.
  GPU-only derived fields and mutable sidecars are represented explicitly rather
  than silently dropped by CPU `GasData` restore.
- **Setup API:** Accept CPU containers plus E7-F1 context and E7-F6-validated
  device request. Validate compatible box/species/particle dimensions before
  conversion. Setup either succeeds with a complete active session or exposes
  no partially usable session.
- **Resource API:** Allocate known fixed-shape resources once or validate
  caller-supplied arrays. Registry keys are typed/internal; concrete kernel
  records remain in their owner modules. Process adapters receive only their
  required view.
- **Lifecycle API:** `begin_step()`/`complete_step()` maintain counters and
  reject re-entry but do not schedule processes. `checkpoint()` is nonterminal;
  `finalize()` is terminal. Both are explicit synchronization/restore boundaries.
- **Restart:** Reconstruct CPU containers and upload into a fresh session.
  E7-F4 preserves opaque mutable RNG/resource state; E7-F8 defines stream
  semantics and exact stochastic restart guarantees.
- **Workflow hooks:** E7-F2/E7-F3 can acquire resource views; E7-F5 consumes the
  lifecycle hooks; E7-F7 extends resources for transport; E7-F8 specializes RNG
  state; E7-F9 validates the complete loop.
- **Compatibility:** Existing direct kernels, conversions, `gpu_context()`, CPU
  runnables, and exports remain unchanged. No hidden transfer or fallback is
  introduced.

## Failure and Atomicity

- Validation and capability failures occur before conversion/allocation where
  possible and do not mutate caller-owned CPU state.
- A failure before any process launch leaves an active session reusable when
  all invariants still hold.
- A failure after launched work may leave resident arrays partially changed;
  mark the session faulted, preserve the original exception, and require
  explicit discard/close. Do not promise rollback or checkpoint uncertain state.
- `close()` is idempotent and never implies restore. `finalize()` is the only
  terminal restore operation. A failed finalization remains observable and may
  be retried only if E7-F6 policy classifies it as pre-transfer/recoverable.

## Security & Compliance

No network, credential, or regulated-data behavior is added. Reject untrusted
dynamic registry keys, cross-device arrays, aliases that violate kernel
contracts, malformed checkpoint versions, and incompatible dimensions before
mutation. Do not deserialize arbitrary Python objects or import process types
from checkpoint strings. Resource sizes derive from validated fixed dimensions,
and allocation errors propagate without fallback. Public exports follow E7-F6;
concrete scratch internals remain non-public.
