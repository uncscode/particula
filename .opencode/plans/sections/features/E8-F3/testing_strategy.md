# Testing Strategy

Tests follow `.opencode/guides/testing_guide.md`: `*_test.py` files live beside
the execution modules, Warp CPU is the installed-Warp baseline, CUDA-specific
capture evidence is optional pass-or-clean-skip, and coverage thresholds are
never lowered.

## Per-Phase Coverage

- **P1 (shipped):** `particula/execution/tests/gpu_resources_test.py` covers
  canonical family/role order, independent shape/byte formulas, dtype item
  sizes, collision/edge capacities, zero dimensions, checked overflow, frozen
  carriers, concrete-only exports, closed-session rejection, and unchanged
  registry state before and after unrelated acquisition. Focused tests pass.
- **P2:** Unit-test every process/control family for complete native records,
  exact schemas/devices, nonaliasing, caller-supplied identity retention,
  omitted allocation, stable reacquisition, and transactional failure.
- **P3:** Unit-test absent/GAS/PARTICLES communication inventories, immutable
  map/configuration identity, diagnostic selections/accounting arrays,
  cross-family overlap rejection, and byte totals.
- **P4:** Unit-test whole-set atomicity under injected failures at early/middle/
  late allocation and RNG initialization, clean retry, exact nested identity
  reuse, capacity/requirement mismatch, and session/resource drift rejection
  before allocation or launch.
- **P5:** Integration-test E8-F2 preparation and repeated uncaptured enqueue with
  the exact set. Spy on `wp.zeros`, `wp.empty`, `wp.array`, registry acquisition,
  payload readback, and synchronization to prove none occur after setup. CUDA
  capture smoke rows pass or cleanly skip without CPU fallback. Validate docs.

## Test Locations and Commands

- Primary unit coverage:
  `particula/execution/tests/gpu_resources_test.py`.
- Prepared/capture integration coverage: E8-F1/E8-F2 adjacent files under
  `particula/execution/tests/`.
- Focused development checks are coverage-disabled assertion checks:

  ```bash
  pytest particula/execution/tests/gpu_resources_test.py -q --no-cov
  pytest particula/execution/tests/ -q -k "resource or prepared or capture"
  ```

  A focused target with `--cov` is invalid comprehensive evidence; inability to
  meet a full-package threshold from that subset is a validation-infrastructure
  issue, not a feature/test failure.
- After focused checks pass, run the full applicable suite and repository
  coverage policy:

  ```bash
  pytest particula/execution/tests/ -q
  .opencode/tools/run_pytest.py
  ```

  The untargeted wrapper supplies repository-configured full-package coverage
  and its normal threshold. If executable changes include `gpu_resources.py`,
  retain its term-missing row in the resident closeout coverage command and the
  aggregate >=80% gate described in the testing guide.
- Run `.opencode/tools/run_linters.py`; run `mkdocs build --strict` for docs.

## Coverage Impact

New accounting, acquisition, rollback, and validation branches increase the
surface of `particula.execution.gpu_resources`. Each production phase therefore
ships branch-focused tests in the same change. Existing checkpoint, RNG,
communication, diagnostics, resident-scheduler, and export tests remain
regression requirements. Default collection and marker behavior must not
change.
