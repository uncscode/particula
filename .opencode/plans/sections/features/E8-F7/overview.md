# Overview

## Problem Statement

Epic H needs actionable CUDA evidence after the captured resident path is
stable. Aggregate elapsed time alone cannot show whether a workload is limited
by Python/driver launch overhead, graph launch, one dominant kernel, poor
occupancy, or memory access. Without a reproducible split, an optimization can
improve one machine-specific number while obscuring the actual bottleneck or
changing a scientific contract.

## Value Proposition

E8-F7 defines representative small and medium resident workloads, records host
launch and device-kernel evidence separately, ranks bounded bottlenecks, and
publishes recommendations tied to one named CUDA machine and software stack.
Raw samples, profiler commands, unavailable rows, and provenance make every
claim auditable without turning profiling into a runtime feature or a portable
performance guarantee.

## P1 Delivered

Issue #1589 delivered host-only profiling-evidence support in
`particula/gpu/tests/profiling_support.py` and hardware-free coverage in
`particula/gpu/tests/profiling_support_test.py`. The implementation freezes the
small `(1, 16, 2)` and medium `(1000, 16, 2)` resident workload matrix, its
executed/unavailable evidence union, bounded canonical JSON, and injected
`.artifacts` raw-report provenance. It does not import Warp, probe CUDA, start
a profiler process, add a public export, or claim that timing evidence was
collected.

## P2 Delivered

Issue #1590 shipped the opt-in native-CUDA collection in
`particula/gpu/tests/benchmark_test.py`. It publishes four separately
schema-valid artifacts for prepared-uncaptured and captured-replay host-launch
and synchronized-elapsed measurements, plus an explicit mode/method manifest
under `.artifacts/benchmarks/profiling/`. Each executed row retains a
replay-count-major raw JSON sample report and checksum provenance under the
controlled `profiling/raw/` root; unavailable CUDA/capture prerequisites instead
publish complete unavailable rows for both frozen workloads without timing or
CPU/Warp-CPU fallback.

The qualified fixture now snapshots mutable resident primary arrays and acquired
continuation sidecars, including RNG state, in
`particula/execution/tests/resident_benchmark_cuda_support.py`. Its private
reset drains, restores pre-bound same-device storage, drains again, and validates
the retained identities before the next batch; it never rebuilds or recaptures.

## P3 Delivered

Issue #1591 delivered test-only, closed native-CUDA Nsight evidence collection.
`particula/gpu/tests/profiling_support.py` now owns strict versioned evidence
records and parsers, bounded `shell=False` subprocess orchestration, exact tool
qualification, contained export handling, and explicit unavailable/failed
outcomes. `profiling_workload_runner.py` accepts only the fixed small
captured-replay worker argv before lazy CUDA imports; `profiling_smoke_test.py`
adds an opt-in CUDA smoke path for `nsys` and `ncu`.

Hardware-free coverage exercises parsing, commands, failure bounds, path safety,
worker argv/call order, and smoke composition. Real collection remains opt-in
and CUDA-only, with no CPU or Warp-CPU fallback. This phase adds no public API,
user-facing documentation, production scheduler behavior, benchmark result, or
recommendation.

## User Stories

- As a performance engineer, I want host launch and device execution costs
  separated so that graph-capture benefits are not confused with kernel speed.
- As a kernel maintainer, I want dominant kernels linked to occupancy and
  memory-access evidence so that follow-up work targets measured constraints.
- As a simulation user, I want recommendations bounded to the measured device,
  workload, versions, and method so that I do not mistake one CUDA profile for
  universal performance guidance.
