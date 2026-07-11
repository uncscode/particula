## Open Questions

Status: reviewed and answered on 2026-07-11.

## Resolved Decisions

- Create a runnable `.py` example now; do not block issue #1263 on creating a
  paired notebook artifact.
- Issue #1263 shipped the runnable `.py` source first, and issue #1264 then
  published the paired notebook, Dynamics index link, and one feature-doc
  cross-link without broadening the docs rewrite scope.
- Use a fast CPU-only water latent-heat setup that produces a non-zero
  temperature/energy diagnostic. Keep the setup deterministic and avoid GPU
  dependencies because GPU latent-heat condensation is explicitly downstream.
- Prefer printed tabular diagnostics for reliable CI/notebook execution. Add a
  plot only if it is lightweight and does not become the validation oracle.
- Keep docs-surface protection in the existing example test module: assert the
  notebook path exists, the Dynamics index points to the notebook, the old raw
  command entry stays absent, and the feature page keeps exactly one direct
  latent-heat example link.

## Implementation Assumptions

- Treat production code as stable; add tests only if a bug is found while making
  the example runnable.
- E3-F7 should consume the final example path for cross-linking when available.
- The editable source of truth remains
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`; the
  `.ipynb` artifact is published output, not the hand-edited source.
