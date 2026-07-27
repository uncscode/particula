# Overview

**Feature:** E7-F1 — Backend-Selection and Execution-Context API  
**Parent:** [E7](../../epics/E7/vision_problem.md)  
**Scope authority:** issue #1451, Track T1

## Implementation Status

**E7-F1-P1 shipped for issue #1462; P2 shipped for issue #1463.**
`particula.execution` now provides the dependency-neutral typed capability
vocabulary and immutable pure lookup matrix, plus typed request validation and
exact adapter selection in context-local private registries. P2 preserves
opaque Warp device identifiers and canonicalizes only the CPU `"cpu"` spelling.
It deliberately does not execute adapters, probe availability, transfer data,
retry, fall back, define state/result contracts, or publish package exports;
those remain later phases.

## Problem Statement

Particula exposes deterministic CPU `RunnableABC` processes and deliberate
direct Warp kernels, but it has no typed user-facing boundary that selects a
backend, states required capabilities, or describes execution-state ownership.
Callers therefore must know backend-specific container, device, mutation, and
return conventions before any resident simulation can be assembled.

## Value Proposition

Define a backend-neutral execution contract and capability matrix in a separate
execution-context module. CPU remains the independent reference implementation;
later E7 tracks can add GPU adapters, resident state, and explicit fallback
policy without changing process physics or passing Warp state through the
`Aerosol`-specific runnable interface.

## User Stories

- As a simulation user, I want typed backend and device selection so unsupported
  requests fail during validation rather than moving data implicitly.
- As a process maintainer, I want one capability contract for state, mutation,
  and results so CPU and future GPU adapters have comparable semantics.
- As an E7 implementer, I want a stable root dependency that E7-F6 can harden
  and E7-F2, E7-F3, and E7-F4 can implement against independently.
