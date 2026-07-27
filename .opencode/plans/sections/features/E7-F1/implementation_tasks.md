# Implementation Tasks

## Backend API

- [ ] Create `particula/execution.py` without importing `particula.gpu` or Warp.
- [ ] Define typed immutable backend, process, device, and capability values.
- [ ] Implement `CapabilityMatrix.supports()` for pure support queries and
  `CapabilityMatrix.require()` for deterministic preflight validation.
- [ ] Define execution-state/adapter protocols and `ExecutionResult` with
  explicit state identity and mutation metadata. (P3)
- [x] Define frozen `ExecutionRequest` and implement `ExecutionContext`
  construction and exact adapter resolution from an explicit request; reject
  missing registrations and capability mismatches. (P2, #1463)
- [ ] Implement `CPUExecutionAdapter` delegation to `RunnableABC.execute()` and
  preserve exact time-step, substep, exception, and returned-state semantics.
- [x] Keep private context-local registration constrained to typed
  process/backend keys; reject duplicate or malformed registrations before
  changing the registry. (P2, #1463)
- [ ] Add narrow exports to `particula/__init__.py`; do not re-export direct
  kernels, sidecars, or concrete GPU configuration types.

## Validation and Compatibility

- [x] Validate backend/device compatibility before selection and define stable
  validation order in tests. (P2, #1463)
- [ ] Validate finite nonnegative time inputs before adapter execution. (P4)
- [ ] Make unsupported requests fail closed; do not catch execution errors to
  attempt CPU or another device.
- [x] Prove importing and resolving a CPU selection path works when Warp is
  unavailable. (P2, #1463)
- [ ] Verify existing `RunnableABC`, `RunnableSequence`, conversion helpers,
  and direct kernel APIs remain behaviorally and import-surface compatible.
- [ ] Document extension seams for E7-F2/F3 adapters, E7-F4 state, and E7-F6
  availability/fallback policy without implementing those tracks.

## Tooling / Tests

- [x] Add focused P2 unit coverage in `particula/tests/execution_test.py` for
  request/context/registry selection and rejection cases. (P2, #1463)
- [ ] Add `particula/tests/execution_exports_test.py` to lock exact exports and
  optional-dependency behavior in a fresh subprocess.
- [x] Use fake adapters to assert selection identity, zero execute calls, and
  no fallback or transfer behavior. (P2, #1463)
- [ ] Use fake runnables/adapters to assert execution arguments, mutation
  declarations, and error propagation. (P3/P4)
- [ ] Run focused tests with `-Werror`, then repository Ruff and mypy checks;
  maintain at least 80% changed-module coverage without lowering thresholds.
- [ ] Run `mkdocs build --strict` and relevant docs contract tests in P6.
