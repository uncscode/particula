# Success Criteria

## Delivered P1 criteria (issue #1462)

- [x] `particula.execution` provides frozen, validated standard-library-only
  backend/device/process/capability declarations and an immutable capability
  matrix without importing Warp or `particula.gpu`.
- [x] Nonempty capability requests use exact complete-declaration matching;
  separate declarations cannot be silently combined. Empty requirements require
  a declared device/process base.
- [x] `supports()` and `require()` validate request arguments in fixed order,
  remain read-only, and use the specified deterministic unsupported-request
  error.
- [x] Unit coverage includes constructor boundaries, immutable value semantics,
   matrix behavior and non-mutation, and a fresh guarded no-Warp import.

## Delivered P2 criteria (issue #1463)

- [x] `ExecutionRequest` and a separate `ExecutionContext` validate typed
  backend/device/process/requirements selection without executing an adapter.
- [x] CPU selection canonicalizes only `Device(Backend.CPU, "cpu")`; Warp
  native identifiers remain opaque and dependency-neutral.
- [x] Resolution performs request/device validation, capability validation, and
  exactly one exact private lookup in that order; unsupported requests do not
  query the registry.
- [x] Private registrations are context-local, typed, and non-replacing; tests
  prove exact identity selection and rejection/non-mutation behavior.
- [x] P2 tests prove no adapter execution, retry, fallback, transfer, optional
  backend probing, or optional backend import.

## Delivered P3 criteria (issue #1464)

- [x] Internal runtime structural state/adapter protocols, closed mutation
  declarations, opaque backend results, and frozen execution results are
  defined in dependency-neutral `particula.execution`.
- [x] Validation retains original caller state and valid result objects by
  identity, preserves ordered immutable metadata and opaque values, and never
  executes adapters or inspects opaque payloads.
- [x] Tests lock exact enum/result layouts, frozen carriers, stable malformed
  rejection/non-mutation behavior, and separation from P2 selection.
- [x] P3 adds neither public exports nor user-facing documentation.

## Delivered P4 criteria (issue #1465)

- [x] Internal `CPUExecutionState` satisfies the P3 state seam through its
  opaque payload, and `CPUExecutionAdapter` satisfies the P3 adapter seam.
- [x] The adapter exact-type validates state and validates finite nonnegative
  real time plus positive integral substeps before exactly one positional
  runnable dispatch, without coercing caller scalar objects.
- [x] Success retains the original state and aerosol by identity in an empty
  metadata `ExecutionResult` declared with `MutationScope.STATE`; replacement
  aerosols fail after their sole call and runnable exceptions propagate.
- [x] Focused fake, concrete-runnable, and fresh-process tests prove no GPU
   import/conversion/fallback/selection, while focused tests, coverage, Ruff,
   formatting, and mypy pass.

## Delivered P5 criteria (issue #1466)

- [x] `particula.execution.__all__` and top-level `particula` publish exactly
  `Backend`, `Device`, `Process`, `Capability`, `CapabilityRequirements`,
  `CapabilityDeclaration`, `CapabilityMatrix`, `ExecutionRequest`,
  `ExecutionAdapter`, and `ExecutionContext` by identity.
- [x] `ExecutionContext.register_adapter()` is typed, context-local, and
  selection-only, retaining registry validation order, static inspection,
  duplicate/non-mutation behavior, and zero adapter invocation.
- [x] P3/P4 state, mutation, result, validator, and CPU-adapter names, plus GPU
  APIs, remain outside the new package export boundary.
- [x] Public-surface, registration, guarded CPU-only import, and compatibility
  regression coverage locks the published contract.

- [ ] Backend selection is located in a documented, typed, separate execution
  context rather than embedded in strategies, builders, or `RunnableABC`.
- [ ] The capability matrix deterministically accepts every declared supported
  combination and rejects every unsupported combination before mutation.
- [ ] Backend, device, state ownership, mutation, identity, and return behavior
  are published and documented for downstream adapters.
- [x] The internal CPU adapter preserves existing process physics, delegates
  exact time and substep inputs, and retains current `Aerosol` return semantics.
- [ ] No selection or failure path performs hidden CPU/Warp transfer, catches a
  runtime error to retry another backend, or requires Warp for CPU-only import.
- [x] Public exports are deliberate and contract-tested; direct kernels and
   concrete sidecars are not promoted through the new surface. (P5, #1466)
- [ ] E7-F6 can extend policy and E7-F2/F3/F4 can register implementations
  without changing the T1 request, state, or result vocabulary.
- [x] Existing runnable and GPU export compatibility regressions are covered
   unchanged. (P5, #1466)
- [ ] Every phase includes tests, changed execution code has at least 80%
  coverage, Ruff/mypy pass, and no repository threshold is lowered.
- [ ] Backend/device selection behavior and limitations are published and
  `mkdocs build --strict` succeeds.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| User-facing typed selection boundaries | 0 | 1 separate context API | Export tests |
| Implicit fallback/transfer paths | Not centrally constrained | 0 | Negative tests and transfer spies |
| Declared capability matrix cases tested | 0 | 100% of T1 declarations | Parameterized unit tests |
| CPU adapter dispatches per request | N/A | Exactly 1 | Adapter spy tests |
| CPU-only import success without Warp | Ad hoc APIs only | 100% | Fresh-process test |
| Changed-module statement coverage | N/A | >=80% | pytest-cov |
| Existing process physics changed | 0 desired | 0 | Diff and runnable regressions |
