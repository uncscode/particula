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

## User Stories

- As a scientific user, I want captured and uncaptured loops compared with an
  independent CPU oracle so that launch optimization does not change results.
- As a GPU maintainer, I want conservation, diagnostics, communication, and RNG
  checked separately so that aggregate agreement cannot hide a state defect.
- As an operator, I want stale or terminal captured bindings rejected before
  launch so that invalid sessions cannot silently replay or fall back.
