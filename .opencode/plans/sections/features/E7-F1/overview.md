# Overview

**Feature:** E7-F1 — Backend-Selection and Execution-Context API  
**Parent:** [E7](../../epics/E7/vision_problem.md)  
**Scope authority:** issue #1451, Track T1

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
