# Feature Roadmap

This roadmap summarizes the current direction for Particula feature
development. It is a guide for users and contributors, not a fixed release
schedule.

## Current Focus

- **Strategy-based physics systems**: Continue standardizing dynamics modules
  around strategy, builder, and factory APIs.
- **Runnable workflows**: Make condensation, coagulation, wall loss, and related
  processes easier to compose in clear simulation pipelines.
- **Particle and gas data containers**: Complete migration from legacy facades
  toward explicit `ParticleData` and `GasData` containers.
- **GPU-backed particle-resolved simulations**: Move the existing Warp kernels
  from tested lower-level APIs toward documented, high-level simulation
  workflows.
- **Documentation and examples**: Expand practical examples that connect feature
  guides, theory pages, and runnable notebooks.

## Planned Improvements

### Dynamics Systems

- Add more high-level examples for combining condensation, coagulation, and wall
  loss in shared time-stepping loops.
- Improve guidance for choosing distribution types across discrete,
  continuous-PDF, and particle-resolved simulations.
- Continue exposing new physics models through consistent builders and
  factories.

### Data Model Migration

- Reduce reliance on legacy `ParticleRepresentation` and `GasSpecies` facade
  patterns where newer data containers provide clearer state management.
- Keep migration documentation up to date as APIs stabilize.
- Identify remaining compatibility layers that can be simplified before a
  stable major release.
- Track details in the
  [data-oriented design and GPU roadmap](data-oriented-gpu.md).

### GPU Acceleration

- Integrate existing Warp condensation and Brownian coagulation kernels into
  higher-level user workflows.
- Add documented examples for GPU-resident particle simulations that avoid
  repeated CPU/GPU transfers.
- Define parity, performance, and fallback expectations for CPU, Warp CPU, and
  CUDA execution.
- Track details in the
  [data-oriented design and GPU roadmap](data-oriented-gpu.md).

### Examples and Education

- Add more end-to-end chamber simulation examples.
- Expand notebook coverage for feature systems that currently have only API
  documentation.
- Improve links between examples, feature guides, theory pages, and API
  reference material.

## Roadmap Artifacts

This folder can also hold supporting artifacts for planning and examples, such
as design notes, example outputs, prototype workflows, and milestone-specific
pages.

- [Data-Oriented Design and GPU Roadmap](data-oriented-gpu.md)

## Contribution Opportunities

- Add a new physics strategy with builder, factory, tests, and documentation.
- Convert lower-level utilities into feature-level examples that show complete
  workflows.
- Improve theory pages with citations, assumptions, and model limitations.
- Report missing documentation or unclear APIs in GitHub Issues or Discussions.

## How to Propose Changes

For small fixes, open a pull request directly. For larger features, start with a
GitHub Discussion so maintainers and users can align on scope, API shape, and
testing expectations before implementation.
