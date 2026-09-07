# Testing Strategy

Every phase ships tests in the same change. Coverage thresholds are never
lowered. Warp CPU covers hardware-free import, schema, lifecycle, rejection,
and documentation contracts; CUDA graph execution is optional local evidence
that must pass or cleanly skip and never falls back to CPU.

## Per-Phase Checks

- **P1 (implemented, #1595):**
  `particula/tests/gpu_resident_graph_capture_docs_test.py` covers lazy imports,
  force-disabled and capability-unavailable paths, propagated failures, source
  ordering, result observations, lifecycle/teardown, limitations, and the
  Examples-index link. It includes hardware-free contracts and an optional
  native-CUDA smoke row for the one-capture/two-replay lifecycle; no CPU or
  Warp-CPU capture substitution is permitted.
- **P2:** Extend the documentation contract test to require every recapture
  trigger, mutable-value non-trigger, lifecycle state, limitation, failure
  procedure, and reproduction command. Validate internal links.
- **P3:** Add `particula/tests/gpu_graph_capture_closeout_docs_test.py` to parse
  the closeout schema, map every E8 success criterion to evidence, require
  literal results and target derivation, distinguish required from optional
  rows, and reject a Shipped status with missing/failed evidence.
- **P4:** Validate roadmap/index/AGENTS cross-references, T7/T8 reconciliation,
  E8 plan consistency, and the strict MkDocs build.

## Focused Development Checks (Coverage Disabled)

Use direct pytest for affected assertions:

```bash
pytest particula/tests/gpu_resident_graph_capture_docs_test.py \
  particula/tests/gpu_graph_capture_closeout_docs_test.py -q
pytest particula/execution/tests/ -q
```

These targeted commands intentionally provide no coverage evidence. A focused
target combined with coverage is invalid evidence under repository policy; it
is a validation-infrastructure mistake, not a feature test failure.

## Full Validation and Closeout Evidence

After focused checks pass, run sequentially and retain literal output:

```bash
.opencode/tools/run_linters.py
.opencode/tools/run_pytest.py
mkdocs build --strict
```

Before additional changed-module coverage, derive the exact executable module
list from E8-F1--E8-F7 implementation records and the final diff. Run the
applicable full execution suite with repository-configured coverage and retain
per-target term-missing rows plus the normal threshold; documentation-only
files are never coverage targets. Do not substitute focused-target coverage for
the untargeted full-package run.

Record optional CUDA commands separately with device qualification. A clean
skip is availability evidence only and cannot satisfy an exit criterion that
requires measured CUDA capture, scaling, memory, or profiling results.
