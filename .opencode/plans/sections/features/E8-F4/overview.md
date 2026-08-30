# Overview

## Problem Statement

The resident scheduler currently validates and dispatches every timestep from
Python. Existing CUDA graph tests stop at a single condensation call and prove
that public wrappers contain host readbacks that cannot be captured. E8-F1,
E8-F2, and E8-F3 define the compatibility lifecycle, prepared enqueue plan, and
identity-stable resource set, but no owner yet captures that complete prepared
sequence, guards replay, or invalidates the opaque graph when its binding drifts.

## Value Proposition

E8-F4 turns the prepared resident plan into an executable, concrete-only Warp
graph boundary. Setup captures the authoritative fixed twelve-node sequence
once on a qualified CUDA device; replay checks exact lifecycle and compatibility
metadata before launching the same opaque graph handle. Structural drift,
terminal resident state, and post-launch failures invalidate or fault the graph
without hidden recapture, CPU fallback, transfer, synchronization, or retry.
The feature also establishes multi-timestep CPU, uncaptured Warp, and captured
CUDA evidence for the same process configuration.

## User Stories

- As a GPU simulation operator, I want a prepared resident timestep captured
  once and replayed safely so repeated fixed-shape timesteps avoid Python launch
  orchestration without changing physics.
- As a library maintainer, I want exact pre-launch compatibility checks and
  deterministic invalidation reasons so stale graphs never launch against
  replaced arrays, schedules, maps, or devices.
- As a scientific reviewer, I want captured results compared with CPU and
  uncaptured GPU references so graph execution is supported by numerical,
  conservation, RNG-lifecycle, and failure-state evidence.
