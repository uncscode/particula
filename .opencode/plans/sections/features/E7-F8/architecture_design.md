# Architecture Design

## High-Level Design

P1 is a direct-only, caller-owned boundary. `StreamRegistry` retains a canonical
two-process tuple of state arrays by identity and keeps key/descriptor/lane
registration host-only. `initialize()` is the sole optional-dependency operation.

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
- **Delivered P2 dispatch/checkpoint boundary:** The concrete resident Brownian
  adapter validates exact session/resource/collision/RNG bindings and dispatches
  with literal `initialize_rng=False`. It has no reset, transfer, synchronization,
  or fallback path. Checkpoint and finalize fail closed before payload work when
  the resident sidecar is published; RNG metadata and words are not serialized,
  and restart continuation is unsupported.
- **Deferred integration:** P2 does not add wall-loss resources, generic
  reset/inspection APIs, box-invariance machinery, public exports, or RNG
  persistence/restart continuation.

## Security & Compliance

This is a simulation RNG, not a cryptographic primitive. Reject duplicate or
unbounded IDs, malformed checkpoint versions, unknown process namespaces,
cross-device arrays, shape/dtype mismatches, and arbitrary deserialized objects.
Resource sizes remain bounded by validated box count. No network, credentials,
silent fallback, executable checkpoint payload, or dynamic import is introduced.
