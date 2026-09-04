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
increments. P1 qualifies an exact READY binding and P2 captures its frozen
twelve-operation sequence. P3 is delivered: an authentic P2-issued captured
record is provenance-checked, revalidated against its exact binding, then
launched exactly once under one resident token. Replay rejects identity,
lifecycle, device, and duration drift before token entry; it permits mutable
pinned payload and resident RNG-word changes without reseeding. Native launch
or completion failures use writer-capable no-rollback fault semantics. Later
phases own teardown and multi-timestep evidence.

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
