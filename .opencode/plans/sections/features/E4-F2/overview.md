# Overview

**Problem Statement:** GPU condensation currently assumes unit activity and a
static per-species surface tension. It therefore cannot reproduce the selected
CPU activity and composition-dependent Kelvin physics required by issue #1272.

**Value Proposition:** Add fixed-shape, fp64 Warp physics for ideal and kappa
activity plus selected effective surface-tension modes, while retaining the
existing per-species surface input and explicit CPU/GPU boundary.

**User Stories:**
- As a modeler, I want GPU particle-side vapor pressure to include supported
  activity and Kelvin effects so CPU and GPU condensation fixtures agree.
- As a GPU caller, I want numeric model configuration and early validation so
  unsupported physics never silently produces plausible results.
- As a maintainer, I want deferred models such as BAT documented as CPU-only.

This epic-linked feature follows E4-F1 and supplies physics used by E4-F4 after
the parallel E4-F3 substep work converges.
