## Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Create a runnable `.py` example and pair it with a notebook if the surrounding
  examples directory uses the paired workflow. Repository guidance prefers
  editing paired `.py` files and syncing notebooks when a notebook artifact is
  present.
- Link the new example from both the Dynamics examples index and
  `docs/Features/condensation_strategy_system.md` if that feature page exists at
  implementation time. The index provides discovery; the feature page provides
  conceptual context.
- Use a fast CPU-only water latent-heat setup that produces a non-zero
  temperature/energy diagnostic. Keep the setup deterministic and avoid GPU
  dependencies because GPU latent-heat condensation is explicitly downstream.
- Prefer printed tabular diagnostics for reliable CI/notebook execution. Add a
  plot only if it is lightweight and does not become the validation oracle.

## Implementation Assumptions

- Treat production code as stable; add tests only if a bug is found while making
  the example runnable.
- E3-F7 should consume the final example path for cross-linking when available.
