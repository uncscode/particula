# Overview

## Problem Statement

Issue #1451 Track T6 requires the backend-selection layer to behave predictably
when Warp, a requested device, or a requested GPU process is unavailable. Today
the direct GPU surface uses several import and runtime error patterns, while no
single policy defines when CPU fallback is permitted, what users may import, or
which APIs are stable.

## Value Proposition

E7-F6 gives E7 adapters and sessions one typed error taxonomy, fail-closed
availability checks, and an opt-in CPU transition that is observable and occurs
only at a CPU-authoritative pre-upload or caller-asserted restored boundary. It
also freezes a
narrow public execution surface while marking low-level `particula.gpu.*` APIs
experimental until backend selection and full-loop validation are complete.

## Implementation Status

E7-F6-P1 was implemented for issue #1500 in commit `d1a000769`. It provides a
dependency-neutral, direct-import-only capability-error taxonomy in
`particula/execution/errors.py` with co-located contract tests.

E7-F6-P2 was implemented for issue #1501. The concrete, direct-import-only
`particula.execution.availability` resolver validates valid E7-F1 request and
matrix metadata fail-closed in recognition, declarations, runtime, device, and
state order. It returns a frozen request-only decision, keeps Warp native
identifiers opaque with lazy optional runtime import, and remains unexported.
Co-located tests cover its contract. Package exports remain deferred.

E7-F6-P3 was implemented for issue #1502. The direct-import-only
`particula.execution.fallback` module is default-deny: only explicit CPU policy
may dispatch once for five eligible availability/support reasons and only with
exact CPU-authoritative state. It retains requested/selected backend and
capability-reason provenance separately from unchanged native result metadata,
does not recover or move state, and is covered by focused contract tests and
feature documentation.

## User Stories

- As a simulation author, I want unsupported GPU requests to fail with a clear,
  inspectable capability error so I can correct configuration safely.
- As an operator, I want CPU fallback to require an explicit request so a long
  resident run never moves data or changes backend silently.
- As a library consumer, I want documented stable and experimental import paths
  so upgrades do not unexpectedly bind me to internal sidecars or kernels.
