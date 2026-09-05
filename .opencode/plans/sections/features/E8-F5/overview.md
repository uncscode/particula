# Overview

## Problem Statement

E8's captured resident timestep needs scientific and lifecycle evidence that
goes beyond a successful graph launch. The same fixed process configuration must
produce compatible results through the independent CPU reference, uncaptured
Warp execution, and captured CUDA replay while preserving communication,
diagnostics, conservation, persistent RNG, and fail-closed lifecycle behavior.

## Value Proposition

E8-F5 supplies a reusable three-way validation matrix. It makes graph capture an
evidence-backed optimization rather than a separate scientific execution mode,
detects replay-only state defects, and gives downstream benchmark, memory, and
documentation tracks a stable correctness gate.

## Implementation Status

E8-F5-P1 shipped for issue #1575 as test-only support in
`particula/execution/tests/captured_full_loop_test.py`. It provides an immutable
deterministic two-box scenario and an independent NumPy full-loop oracle for
closed gas communication, prescribed volume evolution, dilution, derived
saturation, and inventory diagnostics. No production API, module, scheduler,
graph-capture, or resident-session architecture changed.

E8-F5-P2 shipped for issue #1576 in the same test module. It adds test-only
uncaptured Warp-CPU READY prepared-path multi-timestep parity and independent
conservation evidence, including forbidden-work spies and zero-duration
coverage. It changes no production module, API, scheduler behavior, or capture
and replay behavior.

E8-F5-P3 shipped for issue #1577 in
`particula/execution/tests/captured_full_loop_test.py` only. It adds optional
native-CUDA captured resident-loop evidence for separate closed-map GAS and
PARTICLES families across active, prescribed-volume, and no-work cases. The
matrix compares replayed CUDA state and diagnostic/work-buffer snapshots with
an independently enqueued Warp-CPU binding, retains opaque CUDA device strings,
rejects unqualified candidates before capture/guard entry, and forbids visible
host setup, allocation, transfer, readback, synchronization, or resource work
during replay. No production API, export, user documentation, example, or
architecture changed.

## User Stories

- As a scientific user, I want captured and uncaptured loops compared with an
  independent CPU oracle so that launch optimization does not change results.
- As a GPU maintainer, I want conservation, diagnostics, communication, and RNG
  checked separately so that aggregate agreement cannot hide a state defect.
- As an operator, I want stale or terminal captured bindings rejected before
  launch so that invalid sessions cannot silently replay or fall back.
