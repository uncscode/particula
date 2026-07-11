## Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Create a runnable `.py` example now; do not block issue #1263 on creating a
  paired notebook artifact.
- Leave Dynamics examples index wiring and any feature-doc cross-linking out of
  issue #1263. The shipped scope is the runnable example plus focused test
  coverage, with discoverability follow-up deferred.
- Use a fast CPU-only water latent-heat setup that produces a non-zero
  temperature/energy diagnostic. Keep the setup deterministic and avoid GPU
  dependencies because GPU latent-heat condensation is explicitly downstream.
- Prefer printed tabular diagnostics for reliable CI/notebook execution. Add a
  plot only if it is lightweight and does not become the validation oracle.

## Implementation Assumptions

- Treat production code as stable; add tests only if a bug is found while making
  the example runnable.
- E3-F7 should consume the final example path for cross-linking when available.
