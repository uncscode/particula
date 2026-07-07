# E2-F9 Testing Strategy

## Documentation Validation

- Run `mkdocs build --strict` (or `python -m mkdocs build --strict`) as the
  authoritative docs validation command for the issue #1222 documentation work.
- Confirm the new feature guide is linked from `docs/Features/index.md` and any
  supporting docs index updates.
- Verify docs import snippets use public API paths and current object names.

Target docs paths for this feature are
`docs/Features/data-containers-and-gpu-foundations.md`,
`docs/Features/index.md`, `docs/Features/particle-data-migration.md`, and any
supporting discoverability updates such as `docs/index.md`.

## Example Validation

- No example validation is required for issue #1222 because no `docs/Examples/`
  files were added in P1.
- Example smoke-running, Warp guards, and notebook sync/execution remain P2
  validation requirements.

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
python -m mkdocs build --strict
python docs/Examples/data_containers_and_gpu_foundations.py
python3 .opencode/tools/validate_notebook.py \
  docs/Examples/data_containers_and_gpu_foundations.ipynb --sync
python3 .opencode/tools/run_notebook.py \
  docs/Examples/data_containers_and_gpu_foundations.ipynb
```

## Non-Goals for Testing

- Do not add heavy GPU benchmarks or require CUDA in CI for this docs track.
- Do not introduce standalone testing-only phases; validation ships alongside
  each documentation/example phase.
