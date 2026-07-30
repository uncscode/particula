# Testing Strategy

Coverage thresholds remain unchanged and changed execution modules must meet at
least 80% coverage. Tests use `*_test.py`, ship with each phase, and run without
requiring CUDA; Warp CPU is used when Warp is installed and CUDA rows skip
cleanly when unavailable.

## Per-Phase Coverage

- **P1 (implemented, issue #1500):**
  `particula/execution/tests/errors_test.py` locks down the exact direct-module
  `__all__` and eight reason codes; root construction; category hierarchy; each
  concrete error's fixed reason, omitted/complete fields, and exact rendering;
  invalid types; ordinary exception chaining; and a subprocess import with
  Warp and `particula.gpu` imports blocked. The focused module coverage command
  is `pytest particula/execution/tests/errors_test.py -q -Werror
  --cov=particula.execution.errors --cov-report=term-missing`.
- **P2 (implemented, issue #1501):**
  `particula/execution/tests/availability_test.py` covers canonical CPU and
  injected-provider success, frozen request identity, opaque lazy Warp native
  identifiers, missing runtime, unavailable/unknown device, unsupported
  process/capability, malformed registries and status results, exception
  mappings, state validation, and exact short-circuit order. Local fakes and a
  subprocess import guard keep the suite independent of Warp/CUDA and verify no
  adapter, transfer, synchronization, mutation, or kernel-launch seam is used.
- **P3 (implemented, issue #1502):**
  `particula/execution/tests/fallback_test.py` uses local typed context, state,
  adapter, payload, movement, and lifecycle fakes without Warp/CUDA. It covers
  all five eligible reasons; identical default-deny re-raises; exactly one
  canonical CPU lookup and dispatch; both accepted authority boundaries;
  rejection of absent/resident/uploaded/mutated state before lookup; immutable
  fallback provenance with unchanged native `ExecutionResult.metadata`; blocked
  optional imports; and adapter/result-validation failures that propagate with
  no retry or recovery.
- **P4 (implemented, issue #1505):**
  `particula/execution/tests/exports_test.py` asserts the exact ordered
  26-name `particula.execution.__all__`, identity-preserving top-level
  re-exports, denied concrete names, and a fresh CPU-only subprocess import
  with Warp and `particula.gpu` blocked. Existing direct GPU imports remain
  callable; no import-time experimental warnings were added.
- **P5 (implemented, issue #1504):**
  `particula/execution/tests/fallback_integration_test.py` uses fresh local P1,
  P2, and P3 collaborators plus forbidden-operation ledgers. It verifies P2
  unavailable-device rejection does not validate later state, look up adapters,
  or cross conversion, transfer, synchronization, kernel, mutation,
  checkpoint, or restore seams; it verifies one explicit P3 CPU dispatch retains
  state identity and separate provenance; and it verifies a CPU adapter sentinel
  propagates unchanged without retry or reselection.
- **P6 (implemented, issue #1505):**
  `particula/tests/execution_selection_docs_test.py` validates the published
  stable value and concrete-only boundary, exact closed reason outcomes,
  resolver order and no-movement rule, the AST-parsed/executed public-only
  selection fence, AST-parsed guarded resident pseudocode, and Feature-index,
  roadmap, and guide links. It executes no resident or fallback workflow.
  Validation passed with the focused execution suite (623 tests), documentation
  suite (16 tests), CPU-only package import, and `mkdocs build --strict`.

## Focused Validation

```bash
pytest particula/execution/tests -q -Werror \
  --cov=particula.execution --cov-report=term-missing --cov-fail-under=80
pytest particula/tests/execution_selection_docs_test.py -q -Werror
python -Werror -c "import particula; import particula.execution"
pytest particula/gpu/tests/kernel_exports_test.py \
  particula/gpu/tests/conversion_test.py -q -Werror
ruff check particula/execution particula/__init__.py particula/gpu/__init__.py
mypy particula/execution --ignore-missing-imports
mkdocs build --strict
```

Regression assertions explicitly preserve CPU behavior, direct GPU API
availability, caller object identity, and no mutation on preflight failure.
