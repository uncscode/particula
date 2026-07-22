# Phase Details

## Sequencing

E6-F1 must provide the canonical CPU contract before P1. Complete P1 through
P4 in order; P5 documents only the shipped implementation and parity evidence.

- [x] **E6-F2-P1:** Define scalar and per-box dilution input contract with unit tests
  - Issue: #1395 | Size: S | Status: Completed
  - Delivered: `dilution_step_gpu` is concrete-module-only and validates finite
    nonnegative scalar coefficient/time inputs plus same-device `wp.float64`
    `(n_boxes,)` coefficient metadata. Valid calls return identical containers
    with no launch or caller-state write; per-box values and container state are
    deferred to P3, and package re-export is deferred to P2.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/tests/dilution_test.py`
  - Tests: Signature/import boundary, scalar normalization, per-box identity,
    zero-box/no-write paths, units, and rejection ordering.

- [x] **E6-F2-P2:** Implement fixed-shape particle and gas dilution kernels with unit tests
  - Issue: #1396 | Size: S | Status: Completed
  - Delivered: `dilution_step_gpu` applies `exp(-alpha * time_step)` in place
    to fixed-shape particle and gas concentrations, returns the original
    containers, preserves protected fields, and retains scalar/per-box
    coefficient and zero-time/scalar-zero no-op paths. Only this entry point is
    exported through `particula.gpu.kernels`.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/__init__.py`, `particula/gpu/kernels/tests/dilution_test.py`
  - Boundary: Per-box coefficient-value validation, complete preflight, and
    rollback remain P3 work.

- [x] **E6-F2-P3:** Add atomic entry-point preflight validation with unit tests
  - Issue: #1397 | Size: S | Status: Completed
  - Delivered: Read-only preflight validates coefficient form, time, masses,
    per-box coefficient schema/values, particle concentration, then gas
    concentration. It requires exact same-device `wp.float64` Warp schemas and
    finite nonnegative coefficient/concentration values before every no-op,
    allocation, or launch; rejected calls preserve caller-owned state.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/tests/dilution_test.py`
  - Tests: Invalid type/rank/shape/dtype/device/domain/state, precedence,
    full-state snapshots, and no-allocation/no-launch spies. Rollback after a
    successfully launched kernel failure remains out of scope.

- [x] **E6-F2-P4:** Add CPU and Warp multi-box parity and invariant tests
  - Issue: #1398 | Size: S | Status: Completed
  - Delivered: Test-only independent NumPy-reference finite-step evidence uses
    `rtol=1e-12`, `atol=0` for required Warp CPU scalar/per-box cases and the
    same optional CUDA matrix. It separately proves repeated nonuniform decay,
    exact scalar-zero/zero-time no-ops, container/field identity, protected
    state, and caller-owned per-box coefficient preservation.
  - Files: `particula/gpu/kernels/tests/dilution_test.py`
  - Tests: One/multi-box and multi-species particle/gas fixtures include zero
    cells/inactive slots; CUDA skips cleanly when unavailable. Production API
    and documentation are unchanged.

- [x] **E6-F2-P5:** Update development documentation for direct GPU dilution
  - Issue: #1399 | Size: XS | Status: Completed
  - Delivered: Published the direct import, explicit-transfer, fixed-shape
    concentration-only mutation, complete preflight/no-op, parity, and deferred
    scope contract. Reconciled the foundation guide, GPU roadmap, `AGENTS.md`,
    README, and documentation indexes without changing production API or physics.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, `README.md`,
    documentation indexes, `particula/tests/dilution_docs_test.py`
  - Tests: Hardware-free scoped-content, lazy-export, and local-link/anchor
    regression coverage; focused command is
    `pytest particula/tests/dilution_docs_test.py -q -Werror`.
