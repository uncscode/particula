# Testing Strategy

Every production phase ships self-contained tests in the same change. Coverage
thresholds must never be lowered; changed execution modules must retain at least
80% coverage. Tests use the repository `*_test.py` convention.

## Per-Phase Coverage

- **P1 — capability matrix (completed, #1462):**
  `particula/tests/execution_test.py` covers declaration equality, hashing and
  frozen assignment; constructor and stable-error boundaries; exact declared
  matches, no inferred combinations, empty-base and empty-matrix rules;
  fixed validation order; pure `supports()`/`require()` behavior; and a fresh
  subprocess import with `warp` and `particula.gpu` guarded. This suite is
  CPU-only and does not simulate adapters, transfers, or process physics.
- **P2 — context and validation (completed, #1463):**
  `particula/tests/execution_test.py` uses fake adapters, matrix/lookup spies,
  and guarded subprocess imports to verify typed construction order, canonical
  CPU normalization, opaque Warp preservation, matrix-before-single-lookup
  ordering, identity selection, context-local registration non-mutation, and
  no adapter execution, retry, fallback, transfer, or optional dependency.
- **P3 — state/result contract (completed, #1464):**
  `particula/tests/execution_test.py` tests structural runtime protocols,
  exact closed/frozen value layouts, caller-state/result identity, ordered
  immutable metadata, mutation-declaration boundaries, opaque backend-result
  identity, and stable rejection/non-mutation semantics. It also proves P2
  registration/resolution stays callable-only and does not execute or validate
  P3 adapter results.
- **P4 — CPU adapter (completed, #1465):**
  `particula/tests/execution_test.py` uses recording and hostile fakes plus a
  concrete dilution runnable to verify one exact positional dispatch, state and
  aerosol/result identity, `MutationScope.STATE`, exact NumPy scalar forwarding,
  invalid state/control zero-call preflight, exception propagation, replacement
  aerosol rejection, and normal zero-time dispatch. A fresh subprocess executes
  the adapter while guarding Warp, GPU, and conversion imports.
- **P5 — exports/contracts:** Add
  `particula/tests/execution_exports_test.py`; test exact intended top-level
  symbols, fresh-process imports with Warp blocked, extension registration, and
  unchanged low-level kernel export boundaries. Run existing runnable and GPU
  export regressions.
- **P6 — documentation:** Run documentation contract tests and
  `mkdocs build --strict`; verify links, examples, support wording, and E7
  dependency references.

## Edge and Negative Cases

- Unknown backend/process/capability, CPU paired with a CUDA-only device, empty
  requirements, duplicate registration, invalid state type, non-finite or
  negative time, invalid substep count, adapter exception, and returned identity
  mismatch where the adapter contract requires identity.
- Optional Warp absent; Warp present with CPU device; CUDA remains optional and
  is not needed to validate this backend-neutral/API-only feature.
- Assertions distinguish request-validation failure from execution failure;
  neither path may trigger a transfer or fallback.

## Verification Commands

```bash
pytest particula/tests/execution_test.py -q -Werror
pytest particula/tests/runnable_test.py -q -Werror
pytest particula/tests/execution_test.py -q \
  --cov=particula.execution --cov-report=term-missing --cov-fail-under=80
ruff check particula/execution.py particula/tests/execution_test.py
ruff format --check particula/execution.py particula/tests/execution_test.py
mypy particula/ --ignore-missing-imports
mkdocs build --strict
```

The export regression at `particula/tests/execution_exports_test.py` is P5
verification and is not part of the completed P1/P2 command sets.
