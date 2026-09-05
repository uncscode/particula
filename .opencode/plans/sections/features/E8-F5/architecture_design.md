# Architecture Design

## High-Level Design

One scenario definition produces independent CPU expectations and two exact GPU
bindings. The uncaptured and captured GPU paths consume the same prepared plan,
resource identities, configuration, logical box IDs, and initial state.

```text
fixed scenario + seeds + N timesteps
              |
      +-------+--------------------+
      |                            |
independent CPU/NumPy        prepared resident setup
oracle + inventories               |
      |                 +-----------+-----------+
      |                 |                       |
      |          uncaptured enqueue      capture once/replay N
      |                 |                       |
      +--------- assertion boundary ------------+
                    |
      primary fields + diagnostics + ledgers
      conservation + RNG metadata + lifecycle
```

Synchronization and host readback occur only at explicit assertion boundaries.
Deterministic fields are compared individually with documented tolerances;
conservation and stochastic behavior have separate criteria. Rejection rows
snapshot launch counts and accessible state to prove failure occurs before graph
launch. A post-launch writer failure follows the existing fault/no-rollback rule.

## Data / API / Workflow Changes

- **Data model:** Add test-only immutable scenario, expected-state, and snapshot
  carriers if needed. Production schemas and captured records remain owned by
  E8-F1--E8-F4.
- **API surface:** No public exports. Tests use concrete execution and graph
  capture modules directly.
- **Workflow hooks:** E8-F5 gates downstream scaling, memory, examples, and
  closeout with a recorded three-way validation matrix. CUDA rows are optional
  pass-or-clean-skip evidence; Warp CPU covers uncaptured behavior when installed.

### P1 implementation

Issue #1575 implements the CPU branch only, co-located in
`particula/execution/tests/captured_full_loop_test.py`. Frozen scenario, private
state, and result carriers retain read-only fixture inputs and detached writable
oracle results. The NumPy transition stages closed gas-map debits/credits,
applies prescribed volume evolution and dilution, then derives saturation and
concentration-weighted inventory diagnostics. It neither constructs GPU bindings
nor changes the production execution architecture.

### P2 implementation

Issue #1576 adds the uncaptured Warp-CPU evidence branch in the same test file.
It constructs a scenario-specific READY prepared binding, executes multiple
prepared enqueues, and compares detached field, closed-GAS-work-buffer,
accounting, and diagnostic snapshots with the P1 NumPy oracle and independent
inventory checks. Scoped test spies reject enqueue-time setup, allocation,
transfer, readback, and synchronization; a zero-duration row verifies
write-free preservation. These are tests only, not production-path changes.

### P3 implementation

Issue #1577 adds the optional native-CUDA captured branch in
`particula/execution/tests/captured_full_loop_test.py`. Test-local discovery
retains each Warp-reported CUDA string unchanged and rejects non-string or
non-CUDA values without substituting a device. For each discovered candidate,
the test constructs independent CUDA and Warp-CPU bindings for GAS or
PARTICLES closed-map scenarios, qualifies the exact CUDA binding, captures it,
and compares its replay snapshot with the uncaptured binding. The captured
matrix includes active, prescribed-volume, and no-work scenarios; snapshots
cover primary/derived fields, diagnostics, and family work buffers. Replay is
wrapped in scoped instrumentation that rejects test-visible conversion,
allocation, copy, readback, synchronization, registry acquisition, and capture
resource publication. Qualification rejection is asserted before capture or
guard-token entry. All of this remains test-only; no execution architecture or
production ownership changed.

### P4 implementation

Issue #1578 adds only test evidence. `rng_invariance_test.py` dispatches real
Brownian coagulation and selected wall loss against distinct registry-owned
sidecars, confirms ordinary dispatch does not reseed them, and compares a
selected reset lane with an independently initialized reference. `checkpoint_test.py`
uses a real process binding to checkpoint both advanced streams under schema v4,
restart them into fresh same-device sidecars without initialization, and dispatch
again. `graph_capture_test.py` parameterizes forged, attachment, signature,
lifecycle, and terminal replay rejections with launch/guard instrumentation;
its separate launch-error row establishes the documented fault/revocation/one-
release boundary. `captured_full_loop_test.py` adds optional native-CUDA
aggregate evidence that both scheduled streams advance. Production ownership and
the replay implementation remain unchanged.

## Security & Compliance

There is no network, credential, or persistence change. Validation is fail
closed: graph handles cannot cross session, registry, device, restart, or
terminal lifecycle boundaries. Test diagnostics expose copied numerical state,
not native pointers or opaque handles. No failure path adds fallback, migration,
automatic recapture, retry, or rollback guarantees.

## P5 Reconciliation

Issue #1579 records documentation and validation evidence only. Its strict
MkDocs gate is unavailable, so this architecture remains unchanged and E8-F5 is
not marked shipped.
