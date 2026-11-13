# Graph-capture qualification gates

**Measured evidence: UNSHIPPED/BLOCKED.** No reviewed benchmark/profiling CUDA
artifact, final source revision, or designated qualified device is recorded.
Missing results are unavailable and unmeasured, not zero or passing.

This is the current evidence authority extracted from the 2026-09-07 snapshot
at revision `7b064295037c3c9a137f5dd13c03f4a40da3f11b`. The historical plan closure on 2026-09-27
did not supply measurements. Historical implementation blockers are not current
implementation status. Use the [operator runbook](../../../guide/gpu/resident/graph-capture.md)
for the supported runtime contract.

## Qualification metadata


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

## Production targets


Future producers must inspect the capture capability, enqueue, preallocation,
capture/replay, parity, benchmark, and profiling implementations and their final
executable diff. Exclude tests, documentation, internal plans, and artifacts;
then freeze a sorted, deduplicated repository-relative production-module list.
Use the [operator runbook](../../../guide/gpu/resident/graph-capture.md),
[benchmark record](resident-benchmarks.md), and
[profiling record](graph-capture-profiling.md) to locate the supported
boundaries and their evidence requirements. These are static refresh inputs,
not a live dependency of this record or its test.

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

Documentation and tests are excluded from production targets.

## Required evidence matrix


Only `PASS` with a populated final revision, designated device, required command
output, and applicable valid provenance satisfies a row. `FAILED`, `STALE`,
`INFERRED`, `UNAVAILABLE`, and `CLEAN-SKIP` do not. Supplemental devices are
metadata only.

| ID | Required evidence | Final revision | Designated device | Status | Evidence / blocker |
| --- | --- | --- | --- | --- | --- |
| H1 | Qualified capture/replay with no replay-time allocation, host transfer, or bulk synchronization | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2, C7; A1 |
| H2 | Deterministic rejection or explicit recapture for structural changes | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2; A1 |
| H3 | CPU, uncaptured Warp, and captured CUDA parity plus tight concentration-weighted conservation | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2, C7; A1 |
| H4 | Persistent nonaliasing RNG continuation, reset, and restart behavior | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2; A1 |
| H5 | Co-located tests and configured `>=80%` coverage | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C4; frozen targets and aggregate gate unavailable |
| H6 | 1/10/100/1000-box scaling evidence or explicit unavailable rows | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C8; benchmark artifact unavailable; A1 |
| H7 | Small/medium captured-versus-uncaptured launch-overhead provenance | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C8; benchmark artifact unavailable; A1 |
| H8 | Logical memory model and observed peak-memory comparison | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C8; benchmark artifact unavailable; A1 |
| H9 | Nsight occupancy/memory-access evidence and bounded follow-up | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C9; profiling artifact unavailable; A2 |
| H10 | Example/runbook/limitations and Starlight build result | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C5; final-revision documentation output unavailable |
| H11 | Exact qualification command matrix and literal results before promotion | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | C2–C5, C7–C9; literal results unavailable |

## Collection commands

These commands remain unexecuted **for this qualification record**. Results from
other revisions or devices cannot populate its gates. Historical C1/C6 prose
tests were removed; Starlight C5 now owns documentation validation. C2–C3
are focused checks, C4 owns aggregate coverage, and C7–C9 are opt-in native-CUDA
collection. A clean skip cannot satisfy measured-evidence gates.

The commands below use the current direct tools after retirement of the old
workflow wrappers. This command update supplies no new qualification evidence;
all required outcomes remain unexecuted for this record.

### C2
`pytest particula/execution/tests/graph_capture_test.py particula/execution/tests/captured_full_loop_test.py particula/execution/tests/rng_invariance_test.py particula/execution/tests/checkpoint_test.py -q --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C3
```bash
ruff check particula/ examples_tests/ scripts/ --no-fix
ruff format particula/ examples_tests/ scripts/ --check
mypy particula/ --ignore-missing-imports
```

```text
NOT RUN — output must be pasted verbatim after execution
```

### C4
`pytest --cov=particula --cov-report=term-missing --cov-fail-under=80`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C5
`npm --prefix website run build`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C7
`pytest particula/execution/tests/captured_full_loop_test.py -q -m "warp and cuda" --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C8
`pytest particula/gpu/tests/benchmark_test.py --benchmark -k resident -v -s --no-cov`

```text
NOT RUN — output must be pasted verbatim after execution
```

### C9
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
| A1 | UNAVAILABLE — absent reviewed benchmark CUDA artifact | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| A2 | UNAVAILABLE — absent reviewed profiling CUDA artifact | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |

## Promotion rule

Require H1–H11 PASS on one final revision and designated qualified CUDA device,
reviewed safe artifacts where applicable, literal command outputs, and frozen
production targets with at least 80% aggregate coverage. Otherwise retain
UNSHIPPED/BLOCKED. Validate manifest consistency before changing row statuses;
keep row and command ordering. Supplemental devices are metadata only.
