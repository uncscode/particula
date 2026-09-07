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
- **P2 (implemented, #1596):**
  `particula/tests/gpu_graph_capture_runbook_docs_test.py` hardware-freely
  validates the runbook's ordered recapture triggers, mutable-value
  non-triggers, lifecycle states, limitations, failure procedures,
  reproduction commands, relative links, and stdlib-only imports.
- **P3 (implemented, #1597):**
  `particula/tests/gpu_graph_capture_closeout_docs_test.py` hardware-freely
  validates the closeout schema, maps every E8 success criterion to evidence,
  requires literal results and target derivation, distinguishes required from
  optional rows, and rejects promotion when evidence is missing or failed. The
  committed closeout disposition remains `UNSHIPPED/BLOCKED`.
- **P4 (implemented, #1598; non-promoting):** Reconciled
  roadmap/index/AGENTS cross-references, T7/T8 ownership, and E8 parent/child
  plan consistency. Focused closeout/runbook documentation contracts passed (16
  passed). Active split-plan validation and `mkdocs build --strict` are required
  but unavailable and pending in this worktree. These documentation checks do
  not provide CUDA evidence or promote Epic H.

## Focused Development Checks (Coverage Disabled)

Use direct pytest for affected assertions:

```bash
pytest particula/tests/gpu_graph_capture_closeout_docs_test.py -q --no-cov
pytest particula/tests/gpu_graph_capture_runbook_docs_test.py -q --no-cov
```

These targeted commands intentionally provide no coverage evidence. A focused
target combined with coverage is invalid evidence under repository policy; it
is a validation-infrastructure mistake, not a feature test failure. P4's
hardware-free documentation results do not provide CUDA evidence or promote
Epic H.

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

P3's fail-closed ledger tests are non-promoting evidence. Completion still
requires E8-F2, E8-F6, E8-F7/T7, and all H1--H11 final-revision rows on the one
designated qualified CUDA device.
