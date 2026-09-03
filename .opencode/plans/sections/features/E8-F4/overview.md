# Overview

## Problem Statement

The resident scheduler currently validates and dispatches every timestep from
Python. Existing CUDA graph tests stop at a single condensation call and prove
that public wrappers contain host readbacks that cannot be captured. E8-F1,
E8-F2, and E8-F3 define the compatibility lifecycle, prepared enqueue plan, and
identity-stable resource set, but no owner yet captures that complete prepared
sequence, guards replay, or invalidates the opaque graph when its binding drifts.

## Value Proposition

E8-F4 is building an executable, concrete-only Warp graph boundary in phased
increments. E8-F4-P1 is delivered: it qualifies one exact READY binding,
prepared resident simulation, and published capture resource set, and lazily
retains the adapter-provided native callable vocabulary for a non-CPU Warp
device. Qualification is metadata-only, preserves READY on success and
rejection, and owns no native graph/exec handle or cleanup. Later phases own
native capture, replay, invalidation/teardown, and multi-timestep evidence.

## User Stories

- As a graph-capture implementer, I want an exact prepared binding qualified
  before native capture so later capture code receives frozen identities and a
  validated callable vocabulary without repeating runtime probing.
- As a library maintainer, I want exact pre-launch compatibility checks and
  deterministic invalidation reasons so stale graphs never launch against
  replaced arrays, schedules, maps, or devices.
- As a scientific reviewer, I want captured results compared with CPU and
  uncaptured GPU references so graph execution is supported by numerical,
  conservation, RNG-lifecycle, and failure-state evidence.
