# Architecture Guide

## GPU Module Boundaries

The GPU package keeps a strict separation between transfer, schema, and
kernel-entry responsibilities.

### Transfer boundary

- `particula/gpu/conversion.py` owns explicit CPU↔GPU transfer helpers only.
- It should not absorb launch-time kernel validation or normalization logic.

### Schema boundary

- `particula/gpu/warp_types.py` defines Warp-backed container schemas only.
- It should remain a passive data-shape layer rather than a behavior layer.

### Kernel normalization boundary

- `particula/gpu/kernels/environment.py` owns shared private normalization and
  validation for GPU kernel entry points.
- This module is the common boundary for accepting legacy scalars, direct
  `(n_boxes,)` Warp arrays, or `WarpEnvironmentData` inputs before launch-time
  work.
- Condensation and coagulation should reuse this boundary rather than
  re-implementing environment validation independently.

## Design Intent

- Keep CPU↔GPU transfers explicit.
- Keep Warp container definitions stable and behavior-free.
- Keep cross-entry-point normalization private to `particula/gpu/kernels/`.
- Share validation at kernel boundaries when multiple GPU entry points consume
  the same environment contract.
