# E2-F9 Overview: Foundation docs and examples

## Problem Statement

Feature E2-F9 closes the documentation and runnable-example gap left by the
data-model and GPU-foundation tracks. Issue #1222 (`E2-F9-P1`) published the
canonical foundation guide, and issue #1223 (`E2-F9-P2`) added the shipped
example entrypoint, examples-gallery discoverability, and smoke coverage for
the documented container and transfer flows.

## Value Proposition

This feature now publishes the canonical guide, a validated runnable example
layer, and the roadmap-facing handoff trail for Epic E2. The branch work added
`docs/Features/data-containers-and-gpu-foundations.md`, linked it from
`docs/Features/index.md`, updated
`docs/Features/particle-data-migration.md`, and then added the runnable example
pair at `docs/Examples/data_containers_and_gpu_foundations.py` and
`docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py` plus
`docs/Examples/Data_Containers/index.md` and examples-gallery links. Issue
#1224 (`E2-F9-P3`) completed the roadmap handoff by updating
`docs/Features/Roadmap/data-oriented-gpu.md`,
`docs/Features/Roadmap/warp-autodiff-limitations.md`, and
`docs/Features/Roadmap/index.md` so downstream work starts from the shipped
guide/example baseline. The result should help new users adopt the containers
safely and help planner agents route later roadmap work without rediscovering
current limits or broadening top-level docs/index content unnecessarily.

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
