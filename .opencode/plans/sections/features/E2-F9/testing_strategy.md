# E2-F9 Testing Strategy

## Documentation Validation

- Run markdown link validation or mkdocs build if configured in the repository.
- Confirm new feature and example pages are linked from their indexes.
- Verify docs import snippets use public API paths and current object names.

Target docs paths for this feature are
`docs/Features/data-containers-and-gpu-foundations.md`,
`docs/Features/index.md`, `docs/Examples/index.md`, and the roadmap cross-links
updated in P3.

## Example Validation

- Smoke-run plain Python examples in a default development environment.
- Guard Warp-specific paths with `WARP_AVAILABLE`; absence of Warp should skip
  or print a clear explanatory message, not fail unexpectedly.
- If examples include notebooks, edit paired `.py` files, sync notebooks, and run
  the repository notebook execution tool.

For P2, keep executable example coverage tied to the concrete example targets:
`docs/Examples/data_containers_and_gpu_foundations.py` and, if added,
`docs/Examples/data_containers_and_gpu_foundations.ipynb`.

## Co-located Phase Validation

- P1 should validate docs links and any code snippets added with the guide.
- P2 should validate each example in the same phase that adds it.
- P3 should run final documentation/index validation after all links are wired.

P1 and P3 are valid docs-only exceptions to production test expansion, but they
must still run the relevant docs checks in the same PR. P2 is not complete until
the example script or notebook is actually smoke-tested.

## Suggested Commands

```bash
python docs/Examples/data_containers_and_gpu_foundations.py
python3 .opencode/tool/validate_notebook.py \
  docs/Examples/data_containers_and_gpu_foundations.ipynb --sync
python3 .opencode/tool/run_notebook.py \
  docs/Examples/data_containers_and_gpu_foundations.ipynb
```

## Non-Goals for Testing

- Do not add heavy GPU benchmarks or require CUDA in CI for this docs track.
- Do not introduce standalone testing-only phases; validation ships alongside
  each documentation/example phase.
