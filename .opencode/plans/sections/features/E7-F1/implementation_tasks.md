# Implementation Tasks

## Backend API

- [ ] Create `particula/execution.py` without importing `particula.gpu` or Warp.
- [ ] Define typed immutable backend, process, device, and capability values.
- [ ] Implement `CapabilityMatrix.supports()` for pure support queries and
  `CapabilityMatrix.require()` for deterministic preflight validation.
- [ ] Define `ExecutionRequest`, execution-state/adapter protocols, and
  `ExecutionResult` with explicit state identity and mutation metadata.
- [ ] Implement `ExecutionContext` construction and adapter resolution from an
  explicit request; reject missing registrations and capability mismatches.
- [ ] Implement `CPUExecutionAdapter` delegation to `RunnableABC.execute()` and
  preserve exact time-step, substep, exception, and returned-state semantics.
- [ ] Keep registration constrained to typed process/backend keys; reject
  duplicate or malformed registrations before changing the registry.
- [ ] Add narrow exports to `particula/__init__.py`; do not re-export direct
  kernels, sidecars, or concrete GPU configuration types.

## Validation and Compatibility

- [ ] Validate backend/device compatibility and finite nonnegative time inputs
  before adapter execution; define stable validation order in tests.
- [ ] Make unsupported requests fail closed; do not catch execution errors to
  attempt CPU or another device.
- [ ] Prove importing and using the CPU path works when Warp is unavailable.
- [ ] Verify existing `RunnableABC`, `RunnableSequence`, conversion helpers,
  and direct kernel APIs remain behaviorally and import-surface compatible.
- [ ] Document extension seams for E7-F2/F3 adapters, E7-F4 state, and E7-F6
  availability/fallback policy without implementing those tracks.

## Tooling / Tests

- [ ] Add focused unit coverage in `particula/tests/execution_test.py` for every
  public and private helper, including edge and rejection cases.
- [ ] Add `particula/tests/execution_exports_test.py` to lock exact exports and
  optional-dependency behavior in a fresh subprocess.
- [ ] Use fake runnables/adapters to assert call counts, arguments, identity,
  mutation declarations, error propagation, and no transfer calls.
- [ ] Run focused tests with `-Werror`, then repository Ruff and mypy checks;
  maintain at least 80% changed-module coverage without lowering thresholds.
- [ ] Run `mkdocs build --strict` and relevant docs contract tests in P6.
