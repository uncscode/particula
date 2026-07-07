# E2-F9 Testing Strategy

## Documentation Validation

- Run `mkdocs build --strict` (or `python -m mkdocs build --strict`) as the
  authoritative docs validation command for the E2-F9 documentation work.
- Confirm the new feature guide is linked from `docs/Features/index.md` and any
  supporting docs index updates.
- Verify docs import snippets use public API paths and current object names.

Target docs paths for this feature are
`docs/Features/data-containers-and-gpu-foundations.md`,
`docs/Features/index.md`, `docs/Features/particle-data-migration.md`, and any
supporting discoverability updates such as `docs/Examples/index.md` and
`docs/Examples/Data_Containers/index.md`.

## Example Validation

- Issue #1222 (`E2-F9-P1`) was docs-only, but issue #1223 (`E2-F9-P2`) adds
  required example validation for the published runnable entrypoint.
- Example smoke-running should prove the CPU-only success path, guarded Warp
  behavior, and the documented gas-restore boundary.

Keep executable example coverage tied to the concrete example targets:
`docs/Examples/data_containers_and_gpu_foundations.py`,
`docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py`, and no
notebook unless one is intentionally published later.

## Co-located Phase Validation

- P1 should validate docs links and any code snippets added with the guide.
- P2 should validate each example in the same phase that adds it, including the
  exact top-level command path documented for users.
- P3 should run final documentation/index validation after all links are wired.

P1 and P3 are valid docs-only exceptions to production test expansion, but they
must still run the relevant docs checks in the same PR. P2 is not complete until
the example script is smoke-tested through both targeted pytest coverage and the
published runnable path.

## Suggested Commands

```bash
python -m mkdocs build --strict
python docs/Examples/data_containers_and_gpu_foundations.py
pytest particula/gpu/tests/data_containers_example_test.py -q
pytest particula/gpu/tests/conversion_test.py -q
python3 .opencode/tools/validate_notebook.py \
  docs/Examples/data_containers_and_gpu_foundations.ipynb --sync
python3 .opencode/tools/run_notebook.py \
  docs/Examples/data_containers_and_gpu_foundations.ipynb
```

## Non-Goals for Testing

- Do not add heavy GPU benchmarks or require CUDA in CI for this docs track.
- Do not introduce standalone testing-only phases; validation ships alongside
  each documentation/example phase.
