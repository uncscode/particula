# Implementation Tasks

## Backend API

- [ ] Create `particula/execution.py` without importing `particula.gpu` or Warp.
- [ ] Define typed immutable backend, process, device, and capability values.
- [ ] Implement `CapabilityMatrix.supports()` for pure support queries and
  `CapabilityMatrix.require()` for deterministic preflight validation.
- [x] Define execution-state/adapter protocols and `ExecutionResult` with
  explicit state identity and mutation metadata. (P3, #1464)
- [x] Define frozen `ExecutionRequest` and implement `ExecutionContext`
  construction and exact adapter resolution from an explicit request; reject
  missing registrations and capability mismatches. (P2, #1463)
- [x] Implement internal `CPUExecutionState`/`CPUExecutionAdapter` delegation
  to `RunnableABC.execute()` and preserve exact time-step, substep, exception,
  state, and returned-aerosol semantics. (P4, #1465)
- [x] Keep private context-local registration constrained to typed
  process/backend keys; reject duplicate or malformed registrations before
  changing the registry. (P2, #1463)
- [x] Add narrow exports to `particula/__init__.py`; do not re-export direct
   kernels, sidecars, or concrete GPU configuration types. (P5, #1466)

## Validation and Compatibility

- [x] Validate backend/device compatibility before selection and define stable
  validation order in tests. (P2, #1463)
- [x] Validate exact CPU state, finite nonnegative real time, and positive
  integral substeps before adapter execution. (P4, #1465)
- [ ] Make unsupported requests fail closed; do not catch execution errors to
  attempt CPU or another device.
- [x] Prove importing and resolving a CPU selection path works when Warp is
  unavailable. (P2, #1463)
- [x] Verify existing `RunnableABC`, `RunnableSequence`, and direct kernel APIs
   remain behaviorally and import-surface compatible. (P5, #1466)
- [ ] Document extension seams for E7-F2/F3 adapters, E7-F4 state, and E7-F6
  availability/fallback policy without implementing those tracks.

## Tooling / Tests

- [x] Add focused P2 unit coverage in `particula/tests/execution_test.py` for
  request/context/registry selection and rejection cases. (P2, #1463)
- [x] Add `particula/tests/execution_exports_test.py` to lock exact exports and
   optional-dependency behavior in a fresh subprocess. (P5, #1466)
- [x] Use fake adapters to assert selection identity, zero execute calls, and
  no fallback or transfer behavior. (P2, #1463)
- [x] Use fake adapters to assert P3 structural behavior, opaque identity,
  mutation declarations, rejection non-mutation, and no execution. (P3, #1464)
- [x] Use fake and concrete runnables to assert P4 positional execution
  arguments, identity/error behavior, and no-GPU-import dispatch. (P4, #1465)
- [x] Run P4 focused tests with `-Werror`, execution-module coverage, Ruff, and
  mypy without lowering thresholds. (P4, #1465)
- [ ] Run `mkdocs build --strict` and relevant docs contract tests in P6.
