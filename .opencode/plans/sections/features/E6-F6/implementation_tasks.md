# Implementation Tasks

## CPU Core

- [ ] Define immutable policy configuration and exact precedence resolver in
  `particula/particles/exhaustion.py`.
- [ ] Define fixed-shape plan and diagnostic contracts consumable by E6-F5
  discovery/activation and E6-F7 source records.
- [ ] Implement read-only validation and feasibility planning across all boxes.
- [ ] Implement deterministic resampling selection, moment accounting, and one
  commit boundary without resizing.
- [ ] Implement bounded same-direction representative-volume, raw-weight, and
  source-demand scaling.
- [ ] Export only stable CPU helpers through `particula/particles/__init__.py`.

## Direct Warp

- [ ] Mirror the CPU resolver and plan representation with same-device,
  fixed-shape `wp.int32`/`wp.float64` caller-owned sidecars.
- [ ] Add allocation-stable validation, resampling, scaling, and commit kernels
  in `particula/gpu/kernels/exhaustion.py`.
- [ ] Reject bad shape, dtype, device, values, capacity, controls, or scratch
  before clearing outputs, launching mutation, or changing volume.
- [ ] Keep concrete configuration/scratch APIs out of broad exports unless an
  existing direct-step convention requires them.

## Tooling / Tests

- [ ] Add `particula/particles/tests/exhaustion_test.py` with an independent
  weighted-inventory and distribution-moment oracle.
- [ ] Add `particula/gpu/kernels/tests/exhaustion_test.py` for Warp CPU parity,
  optional CUDA, supplied identity, and invalid-call snapshots.
- [ ] Cover capacity-sufficient no-op, sparse, full, repeated, and demand larger
  than releasable capacity cases for every policy combination.
- [ ] Run focused tests, full fast regressions, Ruff, mypy, and docs validation
  without reducing the repository coverage threshold.
