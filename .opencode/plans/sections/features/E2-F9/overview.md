# E2-F9 Overview: Foundation docs and examples

## Problem Statement

Feature E2-F9 closes the documentation gap left by the data-model and
GPU-foundation tracks. For issue #1222 (`E2-F9-P1`), users and planner agents
could see `ParticleData`, `GasData`, environment transfer helpers, and roadmap
notes in scattered code and roadmap files, but there was no single foundation
guide that explained schemas, shape conventions, transfer caveats, and current
support boundaries without reading source.

## Value Proposition

This feature publishes a navigable foundation layer for Epic E2. The current
branch work for issue #1222 adds the canonical guide at
`docs/Features/data-containers-and-gpu-foundations.md`, links it from
`docs/Features/index.md`, and updates
`docs/Features/particle-data-migration.md` so migration guidance points back to
the canonical contract page. The result should help new users adopt the
containers safely and help planner agents route later roadmap work without
rediscovering current limits.

## User Stories

- As a particula user, I want a concise guide to data-container shapes and GPU
  transfer helpers so that I can start from documented APIs instead of source.
- As a GPU-feature implementer, I want limitations and schema drift documented
  so that I do not assume unsupported multi-box or fully GPU-resident behavior.
- As a planning agent, I want downstream handoff notes linked from the feature
  docs so that Epic B/C/D/E dependencies are visible during future planning.

## Parent Epic Context

- Parent epic: E2, issue #1172, Data-Model and Numerical Foundations v2.
- Sibling dependencies: E2-F2, E2-F3, E2-F4, E2-F5, and E2-F8.
- This track is documentation/example focused and should not introduce new
  container semantics beyond accurately describing shipped behavior and planned
  handoff points.
