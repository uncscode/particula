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
- **P2:** Availability/context tests cover missing Warp, unavailable CUDA,
  unknown device, unsupported process/capability, supported CPU, and exact
  validation order. Fakes replace hardware dependence; no kernel is launched.
- **P3:** Fallback tests cover default rejection, explicit pre-upload CPU
  selection, absent CPU state, active resident state, explicit-restored state,
  and requested/selected/reason result metadata.
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
