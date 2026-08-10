# Architecture Design

## High-Level Design

P1 is a direct-only, caller-owned boundary. `StreamRegistry` retains a canonical
two-process tuple of state arrays by identity and keeps key/descriptor/lane
registration host-only. `initialize_process()` is the optional-dependency
operation used when only one process stream is being initialized.

```text
root seed + exact StreamKey(process, logical box ID)
                         |
            specified FNV-1a 32-bit derivation (host only)
                         |
 StreamRegistry words indexed by registered physical lane
                         |
  initialize(): validate both caller-owned Warp uint32 arrays
                         |
       two canonical `wp.copy` writes; identities retained
```

## Data / API / Workflow Changes

- **Delivered P1 data model:** Direct-only `StreamKey`, `StreamDescriptor`, and
  `StreamRegistry` map exact unique UTF-8 logical box IDs to physical lanes for
  the canonical `coagulation` and `wall_loss` namespaces. The registry retains
  the caller's ordered two-array manifest by identity and exposes immutable
  metadata and lane-indexed derived words.
- **Delivered derivation:** A specified 32-bit FNV-1a payload over schema version,
  root seed, process ID, and logical ID produces initial words without Python
  `hash()`, lane, registry order, capacity, or unrelated IDs. Same-process word
  collisions reject during host-only construction.
- **Delivered initialization boundary:** `initialize()` lazily imports NumPy and
  Warp, fully validates both supplied same-device contiguous `wp.uint32`
  `(n_boxes,)` arrays (including distinctness and nonaliasing), then performs
  canonical coagulation and wall-loss copies. Preflight failures write neither
  buffer; a failure after a successful first copy has no rollback guarantee.
- **Delivered P2 resident lifecycle:** `ResidentStreamMetadata` validates and
  retains P1 root-seed, logical-ID, and lane metadata without importing Warp.
  First compatible `acquire_coagulation()` constructs a P1 registry, initializes
  one same-device coagulation-only `(n_boxes,)` `wp.uint32` array, and pins the
  registry, resource view, and array by identity. Reacquisition neither allocates
  nor reseeds it.
- **Delivered P2 dispatch boundary:** The concrete resident Brownian
  adapter validates exact session/resource/collision/RNG bindings and dispatches
  with literal `initialize_rng=False`. It has no reset, transfer, synchronization,
   or fallback path.
- **Delivered P3 wall-loss lifecycle:** First compatible wall-loss acquisition
  creates and initializes only its candidate `(n_boxes,)` `wp.uint32` sidecar
  from the canonical manifest, using `initialize_process("wall_loss")` when a
  coagulation stream is already published. Manifest initialization may allocate
  temporary peer-process storage only to satisfy the two-process manifest; that
  storage is unpublished, unbound, unexposed, and never reused as a resident
  resource. The published view, bindings, and array are exact
  session/device/schema-bound identities and cannot alias coagulation.
  Reacquisition retains the original advanced array without allocation or
  reseeding.
- **Delivered P3 dispatch boundary:** The resolved scheduler validates and
  supplies the authoritative wall-loss logical-box selection before opening its
  token. The adapter validates that selection and the published wall-loss view,
  then calls one private indexed batch writer with literal `initialize_rng=False`.
  Empty selection avoids lazy kernel resolution. Disabled, prelaunch-skipped,
  zero-time, and valid no-work lanes are not written; a writer-capable failure
  retains the established close-token/fault-session behavior without rollback.
- **Delivered P4 lifecycle boundary:** `StreamRegistry.inspect()` returns frozen
  host-only identity metadata; `initialize_selected()` validates exact-tuple
  selectors and the complete retained two-array schema before writing only the
  selected lanes. `GPUResourceRegistry` exposes inspection and initialization
  only for published process sidecars, in canonical process order, and rejects
  an explicitly unacquired target before any writer. `ResidentSession` exposes
  direct-only `inspect_streams()`, `initialize_streams()`, and deliberate alias
  `reset_streams()` only after exact ACTIVE session/registry/closed-guard
  validation. These paths neither acquire, read back, synchronize, schedule,
   nor persist RNG state.
- **Delivered P6 checkpoint/restart boundary:** Schema-v3 adds an optional,
   frozen continuation carrier after the compatible v1/v2 checkpoint fields.
   Checkpoint preflights canonical published bindings, synchronizes once, and
   bulk-captures at most two little-endian `(n_boxes,)` `uint32` payloads. RNG
   roles are excluded from ordinary resource payloads. Fail-closed restart
   validates v1/v2/v3 before setup/allocation, reconstructs fresh exact-device
   arrays and registry bindings from validated metadata, and publishes restored
   current words without calling normal acquisition or initialization. Explicit
   `initialize_streams()`/`reset_streams()` alone rederive root-seed words;
   normal dispatch and reacquisition retain restored identities and words.
- **Deferred integration:** Full box-invariance machinery, public exports,
   durable persistence/migration, hidden transfer/synchronization, and
   direct-kernel API or physics changes remain deferred.

## Security & Compliance

This is a simulation RNG, not a cryptographic primitive. Reject duplicate or
unbounded IDs, malformed checkpoint versions, unknown process namespaces,
cross-device arrays, shape/dtype mismatches, and arbitrary deserialized objects.
Resource sizes remain bounded by validated box count. No network, credentials,
silent fallback, executable checkpoint payload, or dynamic import is introduced.
