# E2-F9 Overview: Foundation docs and examples

## Problem Statement

Issue #1172 track T9 closes the documentation gap left by the data-model and
GPU-foundation tracks. Users and planner agents can see `ParticleData`,
`GasData`, Warp transfer helpers, and roadmap notes in scattered code and
roadmap files, but there is no single foundation guide that explains schemas,
shape conventions, transfer caveats, current limitations, and downstream
handoff dependencies without reading source.

## Value Proposition

This feature publishes a navigable foundation layer for Epic E2. It ties
together the CPU environment container from E2-F2, GPU environment transfer
helpers from E2-F3, gas/environment boundary decisions from E2-F4,
scalar-to-per-box kernel migration guidance from E2-F5, support boundaries from
E2-F8, and roadmap dependencies for future GPU-resident work. The result should
help new users adopt the containers safely and help planner agents route later
roadmap work without rediscovering current limits.

## User Stories

- As a particula user, I want a concise guide to data-container shapes and GPU
  transfer helpers so that I can start from documented APIs instead of source.
- As a GPU-feature implementer, I want limitations and schema drift documented
  so that I do not assume unsupported multi-box or fully GPU-resident behavior.
- As a planning agent, I want downstream handoff notes linked from the feature
  docs so that Epic B/C/D/E dependencies are visible during future planning.

## Parent Epic Context

- Parent epic: E2, issue #1172, Data-Model and Numerical Foundations v2.
- Sibling dependencies: E2-F2/T2, E2-F3/T3, E2-F4/T4, E2-F5/T5, and E2-F8/T8.
- This track is documentation/example focused and should not introduce new
  container semantics beyond accurately describing shipped behavior and planned
  handoff points.
