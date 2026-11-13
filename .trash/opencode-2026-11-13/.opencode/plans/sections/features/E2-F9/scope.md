# E2-F9 Scope

## In Scope

- Add or update a user-facing foundation guide under `docs/Features/` that
  explains:
  - `ParticleData` and `GasData` schemas.
  - planned `EnvironmentData` ownership boundaries, explicitly marked as not
    implemented if still absent.
  - single-box, multi-box, binned, and particle-resolved shape conventions.
  - CPU/GPU transfer helpers and schema-drift caveats.
  - current limitations and support boundaries from E2-F8.
- Add minimal runnable examples for:
  - constructing or using data containers from documented APIs.
  - guarded Warp transfers with `WARP_AVAILABLE` and explicit CPU/GPU copies.
- Link the new docs/examples from `docs/Features/index.md`,
  `docs/Examples/index.md`, and the roadmap page where appropriate.
- Add downstream handoff notes for future GPU-resident simulation, graph capture,
  autodiff, precision, and environment-state work.
- Validate docs links and example execution paths appropriate for optional Warp.

## Out of Scope

- Implementing new container classes such as `EnvironmentData` if not already
  provided by dependencies.
- Changing public container fields, GPU transfer helper behavior, or kernel
  signatures.
- Adding full end-to-end GPU-resident simulations, graph-capture examples, or
  autodiff workflows beyond roadmap links and handoff notes.
- Reworking mkdocs/Jupytext infrastructure except for required index links.

## Done Signal

Users and planner agents can find foundation docs, examples, current
limitations, and downstream roadmap dependencies without reading source files.
