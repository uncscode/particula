# Architecture Design

## High-Level Design

The benchmark is an evidence consumer around the shipped resident lifecycle,
not a runtime mode. A validated case creates one exact E8-F5 fixture and E8-F3
resource set. Setup and warmup occur outside timed regions. Uncaptured and
captured samples execute the same fixed timestep count, synchronize at explicit
sample boundaries, and serialize independently so execution order cannot hide
differences.

```text
validated BenchmarkCase + qualified CUDA device
          |
          +--> dimension memory model
          |      primary/inactive capacity
          |      + E8-F3 logical resource report
          |      + diagnostics/communication/checkpoint
          |      + labeled future tape projection
          |                |
          |        budget preflight ----------> unavailable row
          v
E8-F5 validated fixture -> E8-F2 prepared loop -> E8-F4 capture
          |                         |                 |
          |                   warmup outside timing   |
          +---------------- uncaptured samples   captured samples
                                      \             /
                                explicit synchronize
                                         |
                         raw samples + summaries + peak memory
                                         |
                         versioned reproducible JSON artifact
```

## Data / API / Workflow Changes

- **Data model:** Add frozen concrete benchmark case, sample summary, memory
  category, observed-memory, and unavailable-row records. Use integer bytes,
  schema versions, canonical ordering, checked products/sums, and explicit
  provenance (`analytical`, `registry_logical`, `observed`, or `projected`).
- **Memory rules:** Primary and fixed inactive-slot capacity are represented by
  resident array shapes; E8-F3 contributes reusable sidecars exactly once.
  Checkpoint host/device copies and tape projections are separate scenarios,
  never silently included in steady-state resident totals.
- **API surface:** Keep helpers concrete/test-support only. Extend the existing
  `pytest ... --benchmark` entry point and artifact environment overrides; add
  no package or top-level exports and no runtime benchmark switch.
- **Workflow hooks:** E8-F1 through E8-F5 supply lifecycle, prepared enqueue,
  resources, replay, and correctness fixtures. E8-F7 consumes published limits;
  E8-F8 consumes timings and memory evidence for profiling/closeout.
- **Timing:** Record per-sample elapsed duration after warmup, exact step count,
  host timer, synchronization method, and derived median/min/max/mean. Never
  compare setup/capture time with replay-only samples unless separately labeled.

## Security & Compliance

No network, credential, permission, or public API change is required. Artifact
paths remain under the controlled `.artifacts/benchmarks` root. Metadata must
exclude device pointers, payload contents, RNG words, environment secrets, and
unbounded exception details. Shapes and byte arithmetic fail closed on invalid
or overflowing values. Missing CUDA, capture, memory probes, or sufficient
budget produces explicit unavailable evidence; it never triggers CPU fallback
or fabricates a passing result.
