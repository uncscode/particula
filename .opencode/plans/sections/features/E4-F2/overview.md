# Overview

**Problem Statement:** GPU condensation currently assumes unit activity and a
static per-species surface tension. It therefore cannot reproduce the selected
CPU activity and composition-dependent Kelvin physics required by issue #1272.

**Delivered in P3 / issue #1289:** `condensation_step_gpu()` now accepts the
frozen, keyword-only `CondensationActivitySurfaceConfig` sidecar. It applies
ideal or kappa activity only to the configured water species and combines it
with refreshed pure vapor pressure and static or composition-weighted Kelvin
surface tension, while retaining the legacy per-species surface input.

**User Stories:**
- As a modeler, I want GPU particle-side vapor pressure to include supported
  activity and Kelvin effects so CPU and GPU condensation fixtures agree.
- As a GPU caller, I want numeric model configuration and atomic pre-launch
  validation so malformed physics never silently produces plausible results or
  mutates my state.
- As a maintainer, I want deferred models such as BAT documented as CPU-only.

This epic-linked feature follows E4-F1 and supplies physics used by E4-F4 after
the parallel E4-F3 substep work converges.
