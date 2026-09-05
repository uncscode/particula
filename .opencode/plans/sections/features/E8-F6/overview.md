# Overview

## Problem Statement

Epic H needs reproducible evidence for its primary performance axis: many
independent boxes executed through the GPU-resident loop. Existing opt-in CUDA
benchmarks cover direct condensation and coagulation, but they do not compare
captured and uncaptured complete resident timesteps or account consistently for
the memory required by primary state, inactive fixed slots, registry-owned
temporaries, diagnostics, communication, checkpoints, and future autodiff tape.

## Value Proposition

E8-F6 extends the existing `--benchmark` surface with a box-first resident
matrix, raw captured-versus-uncaptured timing samples, explicit unavailable
rows, and a reproducible memory-budget report. The report consumes E8-F3's
canonical logical-byte inventory, separates analytical logical bytes from
allocator-observed peak memory, and records enough hardware and command context
for maintainers to reproduce or qualify every claim.

## Delivered P1 Foundation

Issue #1581 delivered the concrete, host-only evidence foundation at
`particula/execution/tests/resident_benchmark_support.py`, with default-
collection tests in `particula/execution/tests/resident_benchmark_support_test.py`.
It validates frozen benchmark cases, results, timing summaries, and complete
caller-provided provenance before any CUDA-facing work; deterministically
round-trips schema-versioned JSON; and atomically writes generic JSON only
beneath a verified `.artifacts` root. It does not execute benchmarks, import or
probe Warp/CUDA, modify resident execution, add exports, or publish evidence.

## Delivered P2 Capture Comparison

Issue #1582 delivered one opt-in, CUDA-only supplemental comparison of an
already prepared uncaptured resident timestep and an already captured native
replay timestep. The host-only support now emits schema-v2 artifacts with two
comparison timing modes, bounded alternating paired warmups/samples, immutable
summaries, and setup/capture provenance excluded from samples and summaries.
It writes only
`.artifacts/benchmarks/resident_capture_comparison.json` through a dedicated
atomic helper.

`particula/execution/tests/resident_benchmark_cuda_support.py` lazily builds,
qualifies, captures, identity-gates, and closes the CUDA-only test binding;
`particula/gpu/tests/benchmark_test.py` supplies the sole `warp`/`cuda` marked
row. The row cleanly skips when CUDA or native capture is unavailable, never
uses CPU or Warp-CPU as a substitute, changes no production API or generic GPU
benchmark artifact, and makes no speed, regression, or performance claim.

## Delivered P3 Box-First Matrix (Issue #1583)

Issue #1583 delivered the host-only 1/10/100/1000-box matrix and its
budget-aware classification before CUDA probing or fixture allocation. Every
row preserves its exact requested and actual dimensions; equality with the
configured budget is eligible and an over-budget row is recorded as
`skipped_budget` without probing availability. Missing CUDA, an unsupported
device, or unavailable native capture produces a structured `unavailable` row
with a reason, never CPU or Warp-CPU fallback.

The P2 CUDA fixture seam now receives exact box, particle, and species
dimensions. The opt-in resident artifact consumer preflights all matrix rows,
memoizes availability across eligible rows, reuses one binding per approved
row, and writes one aggregate artifact only after all rows complete. This is
test-only evidence: P4/P5 byte-category, allocator, and observed-memory work
remain unimplemented, and no production API changed.

## User Stories

- As a performance engineer, I want captured and uncaptured resident loops run
  against identical fixtures so that launch-overhead savings are measurable.
- As a simulation operator, I want box/particle/species cases checked against a
  memory budget before allocation so that oversized rows skip explicitly.
- As an Epic I planner, I want current and projected tape-memory components
  reported separately so that differentiable workloads can be scoped honestly.
