# Epic H graph-capture closeout record

## Scope and current disposition

**Disposition: UNSHIPPED/BLOCKED.** E8-F7/T7 owns profiling and
machine-bounded recommendations. F8-P3 only records closeout evidence; it does
not promote Epic H, change runtime APIs, or duplicate the F8-P1 example or P2
runbook. Epic H remains unpromoted. Missing evidence is unavailable and
unmeasured, not zero or passing evidence.

The committed record has no populated final revision, designated qualified CUDA
device, measurement, or reviewed artifact. Measurement fields remain
unavailable, and reviewed artifact fields remain unavailable, until qualifying
evidence is collected and reviewed.

## Closeout metadata

| Field | Value |
| --- | --- |
| Record date | 2026-09-07 |
| Final source revision | UNAVAILABLE — evidence not collected |
| Python | UNAVAILABLE — evidence not collected |
| Warp | UNAVAILABLE — evidence not collected |
| CUDA driver/runtime | UNAVAILABLE — evidence not collected |
| Warp availability | UNAVAILABLE — evidence not collected |
| Designated qualified CUDA device | UNAVAILABLE — no designated qualified CUDA device |
| Supplemental devices | UNAVAILABLE — evidence not collected |

## Target-derivation ledger

Future producers must inspect the E8-F1 through E8-F7 implementation records
and final executable diff, exclude tests, docs, `.opencode/plans/`, and
`.artifacts/`, then freeze a sorted, deduplicated repository-relative
production-module list. These static refresh inputs are not a live dependency
of this record or its test:

| Track | Inline repository plan path |
| --- | --- |
| E8-F1 | `.opencode/plans/sections/features/E8-F1/implementation_tasks.md` |
| E8-F2 | `.opencode/plans/sections/features/E8-F2/implementation_tasks.md` |
| E8-F3 | `.opencode/plans/sections/features/E8-F3/implementation_tasks.md` |
| E8-F4 | `.opencode/plans/sections/features/E8-F4/implementation_tasks.md` |
| E8-F5 | `.opencode/plans/sections/features/E8-F5/implementation_tasks.md` |
| E8-F6 | `.opencode/plans/sections/features/E8-F6/implementation_tasks.md` |
| E8-F7 | `.opencode/plans/sections/features/E8-F7/implementation_tasks.md` |

Final executable diff: **UNAVAILABLE — final executable diff not collected**.

### Candidate production modules

UNAVAILABLE — final executable diff not collected.

### Frozen production targets

UNAVAILABLE — final executable diff not collected.

Future coverage rows use exactly this format:

`<repository-relative module> | <literal pytest --cov module command> |
<term-missing output> | PASS|FAILED|UNAVAILABLE`

| Coverage record | Requirement | Status |
| --- | --- | --- |
| Target list | Frozen production targets | UNAVAILABLE |
| Aggregate changed-module coverage | `>=80%` | UNAVAILABLE |

The P3 closeout document and its test are excluded from production targets.

## Required evidence matrix

Only `PASS` with a populated final revision, designated device, required command
output, and applicable valid provenance satisfies a row. `FAILED`, `STALE`,
`INFERRED`, `UNAVAILABLE`, and `CLEAN-SKIP` do not. Supplemental devices are
metadata only.

| ID | Required evidence | Final revision | Designated device | Status | Evidence / blocker |
| --- | --- | --- | --- | --- | --- |
| H1 | Qualified capture/replay with no replay-time allocation, host transfer, or bulk synchronization | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2, C6; F2/F3 incomplete; A1 |
| H2 | Deterministic rejection or explicit recapture for structural changes | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2; F2/F3 incomplete; A1 |
| H3 | CPU, uncaptured Warp, and captured CUDA parity plus tight concentration-weighted conservation | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2, C6; F2/F3 incomplete; A1 |
| H4 | Persistent nonaliasing RNG continuation, reset, and restart behavior | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2; F2/F3 incomplete; A1 |
| H5 | Co-located tests and configured `>=80%` coverage | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C4; frozen targets and aggregate gate unavailable |
| H6 | 1/10/100/1000-box scaling evidence or explicit unavailable rows | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C7; F6 artifact unavailable; A1 |
| H7 | Small/medium captured-versus-uncaptured launch-overhead provenance | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C7; F6 artifact unavailable; A1 |
| H8 | Logical memory model and observed peak-memory comparison | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C7; F6 artifact unavailable; A1 |
| H9 | Nsight occupancy/memory-access evidence and bounded follow-up | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C8; F7 artifact unavailable; A2 |
| H10 | Example/runbook/limitations and strict MkDocs result | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C5; final closeout output unavailable |
| H11 | Exact closeout command matrix and literal results before promotion | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C1–C8; literal results unavailable |

## Command ledger

Rows C1–C5 are validation commands. Focused rows are assertion-only, not
coverage evidence; C4 is this record's sole repository coverage command.
Rows C6–C8 are opt-in, availability-only native-CUDA evidence:
pass-or-clean-skip is not required measured evidence and cannot discharge
H1–H9.

### C1
`pytest particula/tests/gpu_graph_capture_closeout_docs_test.py -q --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C2
`pytest particula/execution/tests/graph_capture_test.py particula/execution/tests/captured_full_loop_test.py -q --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C3
`.opencode/tools/run_linters.py`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C4
`.opencode/tools/run_pytest.py`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C5
`mkdocs build --strict`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C6
`pytest particula/execution/tests/captured_full_loop_test.py -q -m "warp and cuda" --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C7
`pytest particula/gpu/tests/benchmark_test.py --benchmark -k resident -v -s --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C8
`pytest particula/gpu/tests/profiling_smoke_test.py --benchmark -q --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

## Artifact provenance rules

Each artifact/manifest ledger record must contain record ID, manifest pointer,
schema/version, final source revision, workload ID, machine/device provenance,
contained relative raw filename, byte size, and lowercase SHA-256. Reject
absolute or traversal paths, latest-path selection, symlinks, raw local reports,
and copied summaries.

| Record ID | Manifest pointer | Schema/version | Final revision | Workload | Provenance | Raw filename | Byte size | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | UNAVAILABLE — absent reviewed F6 CUDA artifact | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| A2 | UNAVAILABLE — absent reviewed F7 CUDA artifact | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |

## Blockers and promotion rule

Absent reviewed F6/F7 CUDA artifacts and outstanding F2/F3 work block this
record. Promotion requires H1–H11 `PASS` on one final revision and designated
qualified CUDA device, reviewed safe artifacts where applicable, all literal
command outputs, and frozen coverage targets/gate. Otherwise retain
`UNSHIPPED/BLOCKED`.

A later evidence update must preserve row and command ordering, validate
manifest consistency before changing statuses, and remain blocked if any
required input is absent or fails. This is a retry-safe, auditable recovery path
that does not mutate runtime state.
