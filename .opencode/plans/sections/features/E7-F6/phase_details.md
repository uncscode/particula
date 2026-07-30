# Phase Details

- [x] **E7-F6-P1:** Define execution capability error taxonomy with unit tests
  - Issue: #1500 | Size: S | Status: Implemented (`d1a000769`)
  - Goal: Added stable typed exceptions with deterministic messages and structured context.
  - Files: `particula/execution/errors.py`, `particula/execution/tests/errors_test.py`
  - Tests: Exact direct-module exports/reason codes, root and concrete fields,
    hierarchy, deterministic messages, invalid types, chaining, and subprocess
    import neutrality with Warp and `particula.gpu` blocked.
  - Scope retained: no package/top-level exports, availability resolution,
    fallback selection, GPU behavior, or user documentation changed.

- [x] **E7-F6-P2:** Resolve backend availability before execution with unit tests
  - Issue: #1501 | Size: S | Status: Implemented
  - Goal: Resolve valid P1 request/matrix metadata fail-closed before adapter
    mutation in recognition, declarations, runtime, device, and state order.
  - Files: `particula/execution/availability.py`,
    `particula/execution/tests/availability_test.py`
  - Tests: Dependency-free provider/import fakes cover canonical CPU,
    opaque-native lazy Warp, typed failure mapping, malformed injected seams,
    exact short-circuit ordering, immutable request-only decisions, and no
    adapter/transfer/synchronization/mutation/launch access.
  - Scope retained: no adapter selection, fallback, package/top-level export,
    transfer, synchronization, or execution-state mutation.

- [x] **E7-F6-P3:** Implement opt-in CPU fallback boundary with unit tests
  - Issue: #1502 | Size: S | Status: Implemented
  - Goal: Provide a default-deny, direct-import-only CPU fallback boundary that
    dispatches only for explicit eligible capability errors and exact
    CPU-authoritative state at pre-upload or caller-asserted restored boundaries.
  - Files: `particula/execution/fallback.py`,
    `particula/execution/tests/fallback_test.py`,
    `docs/Features/data-containers-and-gpu-foundations.md`
  - Tests: All five eligible reasons, default identical re-raise, one lookup and
    one dispatch, authority/boundary rejection before lookup, provenance
    metadata with unchanged native metadata, import neutrality, and propagation
    of adapter/result failures without retry or recovery.

- [x] **E7-F6-P4:** Freeze public exports and experimental API policy with tests
  - Issue: #1505 | Size: S | Status: Implemented
  - Goal: Published the frozen 26-name value surface while retaining concrete
    mechanics and supported experimental low-level GPU APIs without breakage.
  - Evidence: `particula.execution.__all__` and
    `particula/execution/tests/exports_test.py` enforce the ordered 26-name
    surface, identity-preserving top-level re-exports, denied concrete names,
    and CPU-only import neutrality with Warp and `particula.gpu` blocked.

- [x] **E7-F6-P5:** Add no-silent-fallback integration regressions
  - Issue: #1504 | Size: S | Status: Implemented
  - Goal: Prove unavailable-device rejection reaches no execution/movement seam,
    and explicit CPU fallback preserves provenance/state without retry.
  - Files: `particula/execution/tests/fallback_integration_test.py`
  - Tests: Local provider phase logs and P1 snapshots; CPU/GPU adapter and
    conversion/transfer/synchronization/kernel/mutation/checkpoint/restore
    spies; explicit pre-upload CPU fallback provenance/state preservation; and
    unchanged propagation of a terminal CPU adapter sentinel.

- [x] **E7-F6-P6:** Update development documentation
  - Issue: #1505 | Size: XS | Status: Implemented
  - Goal: Documented stable values, concrete-only mechanics, resolver ordering,
    closed fallback outcomes, and explicit caller-owned resident boundaries.
  - Files: `docs/Features/backend_selection.md`, Feature index, foundation and
    roadmap references, and
    `particula/tests/execution_selection_docs_test.py` regression coverage.
  - Tests: `pytest particula/execution/tests -q -Werror --no-cov` (623 passed),
    `pytest particula/tests/execution_selection_docs_test.py -q -Werror --no-cov`
    (16 passed), CPU-only package imports, and `mkdocs build --strict` passed.
    Regression coverage verifies the exact public surface, concrete-only
    boundary, reason outcomes, resolver/no-movement wording, executable
    selection-only fence, guarded resident pseudocode, and resolved links.
  - Scope retained: documentation does not add automatic movement, recovery,
    retry, availability dispatch, or runtime behavior. Transport, detailed RNG
    policy, broad orchestration, and direct-kernel follow-up remain downstream.
