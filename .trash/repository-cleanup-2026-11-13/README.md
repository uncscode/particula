# Root-file cleanup — 2026-11-13

These files are retained unchanged as historical reference, outside the active
repository root. This archive does not establish that old release tasks closed
or that historical benchmark measurements apply to version 0.3.0.

- `todo_fix.md`: the 2026-10-04 conda release-test checklist and original failure
  log. Its external feedstock update, strict parity and successful external
  build were still pending in that record. Their current status was not checked
  during this cleanup. Use `conda/README.md` at the repository root for the
  maintained release-validation procedure.
- `gpu_benchmark_results.json`: the 2026-07-06 root export, missing the
  `benchmark_metadata` required by the current viewer. The maintained artifact
  under `.artifacts/benchmarks/` and declared documentation input under
  `docs/data/` remain in place.

Removed obsolete root configuration:

- `.opencode-preserve.yaml` and `.adw-docs-manifest.yaml`: retired development
  workflow configuration; the project no longer uses that system.
- `jupytext.toml`: notebook-pairing configuration. Maintained documentation now
  uses percent-cell Python sources and saved website results. Jupytext itself
  is still required by the documentation tools and remains a dependency.
