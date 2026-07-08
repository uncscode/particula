# Success Criteria

- A stable direct-kernel import path is selected and regression-tested.
- `condensation_step_gpu` and `coagulation_step_gpu` are importable from the
  documented public path.
- Any top-level exports added to `particula.gpu` are intentionally narrow and
  covered by `__all__` tests.
- A new user can run the quick-start on Warp CPU without reading source code.
- The quick-start demonstrates both condensation and coagulation direct kernel
  usage.
- Documentation preserves explicit CPU/GPU transfer boundaries and does not
  imply hidden synchronization or hidden transfer.
- Missing Warp and missing CUDA cases are explained and tested/skipped cleanly.
- Device mismatch and mixed environment input troubleshooting are documented.
- No high-level backend selector or automatic CPU/GPU dispatch is introduced.

## Verification Checklist

- [ ] Import/export regression tests pass.
- [ ] Docs quick-start smoke tests pass on Warp CPU.
- [ ] No-Warp path is covered by tests or deterministic branch behavior.
- [ ] Relevant GPU kernel tests still pass.
- [ ] Documentation links from feature/roadmap pages are valid.
