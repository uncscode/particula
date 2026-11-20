# Resident benchmark and memory-budget record

## Scope and source of record

The required resident source of record is the machine-generated schema-v3
artifact `.artifacts/benchmarks/resident_capture_comparison.json`. The legacy
`.artifacts/benchmarks/gpu_benchmark_results.json` is coagulation-only and is
not a resident substitute. This report transcribes reviewed resident artifacts
without inference; it does not create benchmark evidence.

## Reproduction command and fixed matrix

Collect the opt-in CUDA/native-capture-only evidence with no CPU or Warp-CPU
fallback:

```bash
pytest particula/gpu/tests/benchmark_test.py --benchmark -k resident -v -s --no-cov
```

The fixed matrix has 1, 10, 100, and 1000 boxes, each shaped `(B, 16, 2)`,
with 100% activity. Its process list is communication, condensation,
coagulation, dilution, wall loss, nucleation, and diagnostics; it uses gas
communication and gas/saturation diagnostics. It uses seed 1582, two warmups,
and three samples. Planning inputs are a 2 GiB budget and configured request
estimates of 64 MiB, 256 MiB, 1 GiB, and 4 GiB for 1, 10, 100, and 1000
boxes, respectively. These estimates are not measured allocator consumption.

## Current evidence status

| Boxes | Status | Timing | Memory | Publication reason |
| --- | --- | --- | --- | --- |
| 1 | unavailable | not measured | not measured | no reviewed schema-v3 resident artifact is checked in |
| 10 | unavailable | not measured | not measured | no reviewed schema-v3 resident artifact is checked in |
| 100 | unavailable | not measured | not measured | no reviewed schema-v3 resident artifact is checked in |
| 1000 | unavailable | not measured | not measured | no reviewed schema-v3 resident artifact is checked in |

All current rows are unavailable because no reviewed schema-v3 resident
artifact is checked in. They are not measured, are not zero, and do not imply
`skipped_budget` or any other artifact-recorded status.

## Timing and memory evidence schema

A future reviewed artifact is transcribed without inference. It records two
alternating, device-synchronized modes:
`prepared_uncaptured_device_synchronized` and
`captured_replay_device_synchronized`. For each mode, timing vocabulary is
count, minimum, median, mean, and p95. Setup and capture provenance are kept
separate and excluded from timing samples.

Required provenance labels are UTC timestamp, Python/platform, Warp, device,
synchronization, signature, seed, warmups, and sample count. All such metadata
and all timing and allocator values are unavailable in this revision.

The byte vocabulary distinguishes analytical logical steady-state categories:
primary state, registry manifest, selected diagnostics, and selected
communication metadata. Inactive capacity attribution is non-additive.
The checkpoint host-copy scenario is distinct from steady-state bytes.
Allocator-observed CUDA default-pool high-water delta is separate from
analytical logical bytes, as is its signed observed-minus-analytical
difference. Allocator readings require method, version, coverage, and machine
context; unavailable readings are not zero.

Projected logical tape scenarios use `timesteps × state_bytes` for full
retention and `ceil(timesteps / interval) × checkpoint_bytes + interval ×
state_bytes` for periodic checkpoints. Autodiff tape is not implemented or
measured, and unknown Epic I overhead is excluded from these projections.

## Supported limitations

This is documentation-only scope. It does not change collection code,
artifacts, APIs, CI policy, lifecycle behavior, or examples. It provides no
CPU fallback and no Warp-CPU capture emulation. It publishes no inferred
measurements, universal speedups, hard performance CI gates, allocator
guarantees, or implemented autodiff storage.
