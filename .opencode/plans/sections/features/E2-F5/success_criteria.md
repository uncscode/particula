# Success Criteria

## Functional Criteria

- Existing scalar GPU API behavior still passes for condensation and coagulation.
- A per-box environment path is implemented and tested for `n_boxes > 1`.
- Uniform per-box environment values reproduce scalar behavior for the same
  inputs within the existing deterministic or statistical tolerances of each
  kernel.
- Environment shape, `n_boxes`, and device mismatches fail with clear
  pre-launch `ValueError` messages.
- The migration path does not move temperature or pressure ownership into
  `GasData`.

## Verification Criteria

- Focused helper tests cover scalar broadcast, explicit environment acceptance,
  wrong-shape rejection, wrong-device rejection, and any chosen
  scalar-plus-environment conflict behavior.
- Condensation tests prove uniform per-box inputs match scalar results and that
  non-uniform multi-box inputs execute through the intended path.
- Coagulation tests preserve scalar compatibility and verify multi-box
  environment validation without introducing flaky assertions.
- Relevant GPU kernel tests and lint checks pass.

## Documentation Criteria

- API docs and docstrings make scalar compatibility, explicit environment
  precedence/conflict rules, and downstream handoff points explicit.
- Later physics kernels can identify one canonical environment-state feed point
  rather than introducing independent scalar migration logic.

## Done Signal

Issue `#1172` track `T5` is complete when scalar callers remain supported,
multi-box environment inputs are validated before launch, and downstream GPU
work can reuse one documented environment feed path.
