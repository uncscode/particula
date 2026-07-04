# E2-F9 Documentation Updates

## New or Updated User Docs

- Add `docs/Features/data-containers-and-gpu-foundations.md` with:
  - purpose and prerequisites.
  - container schema tables.
  - shape-convention tables.
  - GPU transfer helper API summary.
  - limitations/support-boundary table.
  - downstream roadmap handoff notes.
- Update `docs/Features/index.md` to link the guide.
- Cross-link from `docs/Features/particle-data-migration.md` where users need
  migration context.

## Examples

- Add minimal data-container and GPU-transfer examples under `docs/Examples/`.
- Update `docs/Examples/index.md` with the new examples.
- If examples use notebooks, maintain paired `.py` and `.ipynb` files.

## Roadmap and Handoff Docs

- Update `docs/Features/Roadmap/data-oriented-gpu.md` to point to the foundation
  guide and examples.
- Link `docs/Features/Roadmap/warp-autodiff-limitations.md` from the limitations
  section.
- Include future-work notes for environment state, GPU-resident kernels,
  graph capture, autodiff, and precision/mass representation decisions.
