# Phase Details

## E3-F4-P1: Decide low-level GPU kernel import path and public surface with regression tests

Size: S

Depends on: maintainer direction that the direct quick-start path should come
from `particula.gpu.kernels`, plus alignment with `E3-F1` for later
quick-start guidance that mentions persisted `rng_states`.

Shipped outcome: the stable direct-kernel user path is
`particula.gpu.kernels`, the two step functions were not re-exported from
`particula.gpu`, package-level kernel exports were narrowed to those two names,
and focused regression coverage was added to capture the selected path. The
decision stays narrow: direct low-level kernels only, no backend selector, and
no broad exposure of internal Warp launch functions.

Test coverage in this phase:

- `particula/gpu/tests/kernel_exports_test.py` asserts the selected import path
  resolves without launching Warp kernels.
- Negative regression assertions verify `particula.gpu` does not expose
  `condensation_step_gpu` or `coagulation_step_gpu` and does not list them in
  `__all__`.
- Exact `__all__` assertions verify `particula.gpu.kernels` exposes only the
  two supported step functions.

Deliverables:

- Documented import-path decision in implementation docs and plan updates.
- Test coverage proving the selected import path resolves.
- `particula/gpu/kernels/__init__.py` narrowed to the two supported step
  functions only, while `particula/gpu/__init__.py` remained unchanged for
  kernel-step exports.

## E3-F4-P2: Add stable import/export tests for direct condensation and coagulation kernels

Size: S

Depends on: E3-F4-P1 choosing the supported import surface so regression tests
lock the exact public path and rejection boundaries before docs examples depend
on them.

Shipped outcome: `particula/gpu/tests/kernel_exports_test.py` is now the single
canonical contract file for this import surface. The phase added parametrized
positive checks for `coagulation_step_gpu` and `condensation_step_gpu`, locked
the exact `particula.gpu.kernels.__all__` value, added representative negative
package-export assertions for `apply_coagulation_kernel`,
`apply_mass_transfer_kernel`, `condensation_mass_transfer_kernel`, and
`initialize_coagulation_rng_states`, and preserved one concrete-module import
check proving those internal helpers remain importable from their owning
modules. Duplicate package-export coverage was removed from
`particula/gpu/kernels/tests/coagulation_test.py`.

Test coverage in this phase:

- `particula/gpu/tests/kernel_exports_test.py` parametrizes the supported
  imports for both `condensation_step_gpu` and `coagulation_step_gpu`.
- The same file asserts the exact package `__all__` surface and negative
  package-export coverage for representative raw helper names.
- The top-level `particula.gpu` negative contract remains Warp-independent,
  while the Warp-backed `.kernels` imports stay guarded with
  `pytest.importorskip("warp")` for CPU-only friendliness.

Deliverables:

- Centralized regression tests for direct condensation and coagulation imports
  in `particula/gpu/tests/kernel_exports_test.py`.
- Exact `particula.gpu.kernels.__all__` assertions for the shipped public
  module surface.
- Negative assertions preventing accidental package-level exposure of raw
  internal kernel helpers.
- Removal of duplicate package-export policy checks from
  `particula/gpu/kernels/tests/coagulation_test.py`.

## E3-F4-P3: Create runnable direct-kernel quick-start example with explicit transfer helpers

Size: S

Depends on: E3-F4-P1 and E3-F4-P2 settling the supported import path and export
guardrails, plus E3-F1 confirming the repeated-call RNG usage that the
coagulation snippet is allowed to demonstrate.

Shipped outcome: the canonical runnable example now lives at
`docs/Examples/gpu_direct_kernels_quick_start.py`. It builds deterministic
single-box `ParticleData` and `GasData`, supplies explicit vapor pressure for
condensation, gates the Warp branch with `WARP_AVAILABLE`, lazily imports
`warp` plus `particula.gpu.kernels`, defaults to `device="cpu"`, performs one
`condensation_step_gpu(...)` call and one `coagulation_step_gpu(...)` call, and
round-trips gas and particle results explicitly with the published transfer
helpers. The shipped example does not use `gpu_context` or `environment=`; it
keeps scalar `temperature` / `pressure` inputs explicit and demonstrates
caller-owned `rng_states` with `initialize_rng=True` and `rng_seed=41` for the
single coagulation call.

Test coverage in this phase:

- `particula/gpu/tests/gpu_direct_kernels_example_test.py` covers deterministic
  helper builders, the no-Warp message path, import deferral for
  `particula.gpu.kernels`, `main()` / `__main__` execution, subprocess smoke
  coverage for the canonical path, a failure-path assertion that avoids a false
  success summary, and the real Warp CPU path when Warp is available.
- The same test module stubs the lazy kernel branch to assert one condensation
  call uses scalar `temperature` / `pressure` without `environment=`, while one
  coagulation call uses caller-owned `rng_states`, `initialize_rng=True`, and
  `rng_seed=41`.
- Output assertions pin the explicit transfer-helper messaging and canonical
  example path.

Deliverables:

- Canonical quick-start example at
  `docs/Examples/gpu_direct_kernels_quick_start.py`.
- Adjacent smoke coverage at
  `particula/gpu/tests/gpu_direct_kernels_example_test.py`.
- Demonstrated condensation direct call with explicit gas transfer and vapor
  pressure input.
- Demonstrated coagulation direct call with caller-owned `rng_states` allocated
  in the example and seeded via `initialize_rng=True`, `rng_seed=41`.
- No hidden transfer or backend-selection helpers.

## E3-F4-P4: Document troubleshooting and validate docs quick-start behavior

Size: S

Depends on: E3-F4-P3 producing the stable quick-start artifact. Follow any
shared wording updates from E3-F5 after marker/helper policy lands, but do not
let that later wording pass block the example itself.

Shipped outcome: the final troubleshooting contract is now aligned across the
canonical quick-start, `docs/Features/data-containers-and-gpu-foundations.md`,
and `docs/Features/Roadmap/data-oriented-gpu.md`. The published guidance now
points users to `docs/Examples/gpu_direct_kernels_quick_start.py`, states that
supported direct-step imports come from `particula.gpu.kernels`, keeps
top-level `particula.gpu` limited to `WARP_AVAILABLE` and transfer helpers,
documents Warp CPU as the default runnable path with optional CUDA, and makes
the explicit transfer-boundary / lossy-restore expectations user visible. The
phase verifies a new user can follow the CPU-backed Warp path and understand
how missing Warp, device mismatch, mixed `environment=` plus direct inputs, and
manual data ownership boundaries behave.

Test coverage in this phase:

- Re-run `particula/gpu/tests/kernel_exports_test.py` and
  `particula/gpu/tests/gpu_direct_kernels_example_test.py` after documentation
  updates so the documented import path and runnable example stay aligned.
- Keep the transfer-boundary and restore-behavior guidance aligned with the
  existing helper coverage in `particula/gpu/tests/conversion_test.py`,
  including caller-owned gas-name metadata and dropped helper-only GPU state.
- Run focused kernel regression checks in
  `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/coagulation_test.py` when troubleshooting text
  changes example inputs or validation guidance.
- Treat CUDA as optional: verify CUDA branches skip with clear reasons when
  unavailable instead of failing default CI.

Deliverables:

- Troubleshooting for missing Warp, missing CUDA, device mismatch, mixed
  `environment=` plus scalar inputs, and transfer-boundary mistakes.
- Example smoke tests using existing docs-example test patterns.
- Documentation updates linking the quick-start from relevant GPU feature docs,
  including developer-facing GPU roadmap/foundation guidance updates.
- Documentation that explicitly distinguishes `particula.gpu.kernels` imports
  from the top-level `particula.gpu` helper surface and documents lossy restore
  expectations for transfer helpers.
