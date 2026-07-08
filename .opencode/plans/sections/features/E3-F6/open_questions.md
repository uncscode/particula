## Open Questions

- Should the new example always be notebook-paired, or is a runnable `.py` file
  sufficient if docs maintainers prefer a lighter artifact?
- Should `docs/Features/condensation_strategy_system.md` link directly to the
  new example, or is the Dynamics examples index enough discoverability?
- What exact physical setup gives the clearest non-zero latent-heat diagnostic
  while remaining fast and stable for docs execution?
- Should the example include a small plot, or are printed tabular diagnostics
  preferred for reliable CI/notebook execution?

### Assumptions for First Implementation

- Use the existing condensation example convention and create a paired notebook
  unless implementation discovers a repository preference against new notebooks.
- Keep all execution CPU-only and deterministic.
- Treat production code as stable; add tests only if a bug is found while making
  the example runnable.
