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

## User Stories

- As a performance engineer, I want host launch and device execution costs
  separated so that graph-capture benefits are not confused with kernel speed.
- As a kernel maintainer, I want dominant kernels linked to occupancy and
  memory-access evidence so that follow-up work targets measured constraints.
- As a simulation user, I want recommendations bounded to the measured device,
  workload, versions, and method so that I do not mistake one CUDA profile for
  universal performance guidance.
