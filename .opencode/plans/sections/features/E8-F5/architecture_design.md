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

## Security & Compliance

There is no network, credential, or persistence change. Validation is fail
closed: graph handles cannot cross session, registry, device, restart, or
terminal lifecycle boundaries. Test diagnostics expose copied numerical state,
not native pointers or opaque handles. No failure path adds fallback, migration,
automatic recapture, retry, or rollback guarantees.
