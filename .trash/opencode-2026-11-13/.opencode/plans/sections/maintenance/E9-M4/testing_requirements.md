# Testing Requirements

## Policy

Test coverage thresholds must NEVER be lowered. Each phase includes
self-contained tests committed in the same PR as implementation. Test files
use `*_test.py` in adjacent `tests/` directories; composed integration tests
belong in `particula/integration_tests/`.

Follow `.opencode/guides/testing_guide.md` and active configuration rather than
the generic template's stale numeric coverage assertion. Focused file/folder/
node/marker/name selections are coverage-disabled assertion evidence only.
Focused-target coverage is invalid evidence, not a fix failure. After focused
assertions pass, run `.opencode/tools/run_pytest.py` WITHOUT a target for full
applicable repository assertions and configured full-package coverage at the
normal threshold. Do not supply local `coverageSource`, `coverageThreshold`,
raw coverage targets or arbitrary numerical gates. Do not add local `-Werror`.

## Focused phase gates

Run applicable commands sequentially on the implementation revision, recording
literal outcomes. Add new adjacent test files to the inventory when introduced.

```bash
# P1: composition and wall-loss runnable; rerun affected M3 science as needed
pytest particula/tests/runnable_test.py particula/dynamics/tests/wall_loss_runnable_test.py -q --no-cov

# P2: all-gas dilution and protected concrete-only helper imports
pytest particula/dynamics/tests/dilution_test.py particula/dynamics/tests/dilution_runnable_test.py particula/dynamics/tests/dilution_exports_test.py -q --no-cov

# P3: native orchestration and unchanged transaction authority
pytest particula/dynamics/tests/nucleation_runnable_test.py particula/dynamics/nucleation/tests/particle_source_test.py -q --no-cov

# P4: existing CPU adapters, integrated dispatch and failure policy
pytest particula/execution/tests/condensation_adapter_test.py particula/execution/tests/coagulation_adapter_test.py particula/execution/tests/condensation_integration_test.py particula/execution/tests/coagulation_integration_test.py -q --no-cov
pytest particula/execution/tests/errors_test.py particula/execution/tests/fallback_test.py particula/execution/tests/fallback_integration_test.py -q --no-cov

# P5: composition plus explicit CPU wall-loss regressions
pytest particula/integration_tests/ -q --no-cov
pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov

# Retained transfer/export/ownership boundaries
pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py particula/execution/tests/exports_test.py particula/tests/execution_exports_test.py -q --no-cov
pytest particula/execution/tests/gpu_session_test.py particula/execution/tests/graph_capture_test.py -q --no-cov

# Canonical comprehensive gate, only after focused assertions pass
.opencode/tools/run_pytest.py
.opencode/tools/run_linters.py
mypy particula/ --ignore-missing-imports
mkdocs build --strict
```

The adjacent wall-loss suite is excluded by `norecursedirs`; an untargeted run
does not replace its explicit assertion gate. Strict documentation rendering
applies to the bounded developer-doc changes, not early M5 notebook execution.

## Assertion matrix

- Validate non-unit volumes, unequal weights, species-order sentinels, mixed
  partitioning and independent concentration-weighted inventories. Nucleation
  conservation retains `rtol=1e-12, atol=1e-30` where the existing contract uses
  it; other scientific tolerances remain those justified by M3 tests.
- Dilution independently expects exponential decay for every gas lane and
  particle concentration; compare protected masses/environment/metadata
  separately. Reject invalid zero-work state before writes and inject failures
  after earlier writes to exercise restoration and exception chaining.
- Nucleation tests distinguish rejection before commit, failed-attempt atomicity
  and preserved earlier successful commits. Check current gas each equal
  substep, held environment replacement, exhaustion and no-admission cases.
- Use call spies for sequence order, substep duration, CPU exact-once dispatch,
  result identity, errors and absence of transfers/retries/rebinds. Numerical
  adapter integration must also execute real native strategies, not only mocks.
- Keep deterministic agreement, conservation and stochastic aggregate criteria
  separate. Warp CPU is the baseline when installed; CUDA is optional and skips
  cleanly when unavailable, never CPU fallback or cross-device RNG equality.

Missing mandatory evidence blocks M4 closeout and M5 start. Drafting itself
does not run or claim these implementation tests.
