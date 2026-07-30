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
- **P4:** Export tests assert exact `__all__`, approved top-level imports,
  forbidden internals, unchanged direct GPU imports, and a subprocess import
  with Warp blocked. Avoid import-time experimental warnings that would violate
  repository `-Werror` usage.
- **P5:** `fallback_integration_test.py` snapshots caller state and spies on all
  conversion, synchronization, CPU/GPU adapter, and kernel seams. Capability
  rejection calls none; explicit fallback calls only the CPU path; a sentinel
  post-invocation error propagates without retry.
- **P6:** Validate examples/import snippets and run `mkdocs build --strict`.

## Focused Validation

```bash
pytest particula/execution/tests -q -Werror \
  --cov=particula.execution --cov-report=term-missing --cov-fail-under=80
pytest particula/gpu/tests/kernel_exports_test.py \
  particula/gpu/tests/conversion_test.py -q -Werror
ruff check particula/execution particula/__init__.py particula/gpu/__init__.py
mypy particula/execution --ignore-missing-imports
mkdocs build --strict
```

Regression assertions explicitly preserve CPU behavior, direct GPU API
availability, caller object identity, and no mutation on preflight failure.
