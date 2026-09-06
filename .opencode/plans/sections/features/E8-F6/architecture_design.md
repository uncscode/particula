# Architecture Design

## High-Level Design

The benchmark is an evidence consumer around the shipped resident lifecycle,
not a runtime mode. A validated case creates one exact E8-F5 fixture and E8-F3
resource set. Setup and warmup occur outside timed regions. Uncaptured and
captured samples execute the same fixed timestep count, synchronize at explicit
sample boundaries, and serialize independently so execution order cannot hide
differences.

### P1 implementation boundary

The delivered P1 support module is a standard-library-only, concrete test
boundary. Frozen records validate canonical case IDs, capacities, process
ordering, statuses, samples, summaries, and result references on the host.
Callers provide complete metadata, including explicit unavailable/error-safe
Warp and device values. Normalized schema-versioned JSON is deterministic and
deserialization reconstructs validated records. The generic writer checks
containment and symlink safety below an existing `.artifacts` root before
creating directories or temporary files, then writes via fsync and replacement.
This phase neither imports/probes Warp/CUDA nor invokes resident execution.

### P2 implementation boundary

The paired collector performs uncaptured then replay warmups on the continuing
resident/RNG state without synchronization. Each measured operation is bounded
by `clock(); operation(); synchronize(); clock()` and produces its own immutable
raw tuple and deterministic summary. Setup and native capture are timed once as
provenance only, outside warmup and sample collection. Invalid counts reject
before callbacks; the CUDA binding validates retained resident identities before
the collector starts.

CUDA construction and capture are isolated in
`resident_benchmark_cuda_support.py`, whose import does not import Warp or
probe CUDA. Its qualified context exposes zero-argument prepared enqueue and
captured replay callbacks and performs ordered teardown through the established
prepared-loop close path. The opt-in test persists one schema-v2 envelope at the
fixed resident-comparison destination; it neither invokes nor modifies the
generic benchmark writer.

### P3 implementation boundary

The standard-library-only support module builds four immutable box-first cases
(`1`, `10`, `100`, and `1000`) and classifies each before CUDA work. It accepts
only a positive integer requested-case estimate and configured budget, permits
estimate equality, and retains identical requested and actual shapes. A
strictly over-budget case returns `skipped_budget` without invoking availability;
an eligible preconstruction CUDA/device/native-capture absence returns
`unavailable` with a reason. Invalid inputs and malformed availability carriers
raise before callback invocation.

The opt-in consumer preflights every case, memoizes eligibility availability,
forwards exact dimensions into the P2 fixture seam, and appends structured
nonexecution outcomes to the single aggregate artifact. Postconstruction,
capture, timing, result-validation, and cleanup failures remain errors: they do
not become unavailable rows or partial artifacts. No CPU or Warp-CPU fallback,
capacity downscale, allocator accounting, or production API is introduced.

### P4 implementation boundary

The private support module now models resident memory with checked
Python-integer arithmetic and frozen ordered category records. It derives exact
float64/int32 primary-field sizes, accepts E8-F3 only as one validated logical
byte total, and counts only selected caller-owned diagnostic outputs.
Communication selection and inactive fixed-slot capacity are visible,
non-additive attribution, preventing double counting of E8-F3 resources and
full primary capacity.

Checkpoint primary, sidecar, and inspection copies are excluded checkpoint
scenarios. Full-retention and checkpointed tape calculations are checked
projected scenarios with unknown Epic I overhead explicitly excluded. This
host-only implementation imports neither Warp, NumPy, nor `gpu_resources`,
allocates or inspects no device state, serializes no artifact, and changes no
production API.

```text
validated BenchmarkCase + qualified CUDA device
          |
          +--> P3 host preflight
          |      requested estimate vs. budget
          |      equality eligible; exact shapes retained
          |                |
          |        skipped_budget / unavailable row
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
