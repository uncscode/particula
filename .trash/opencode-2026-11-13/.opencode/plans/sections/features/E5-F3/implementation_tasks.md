# Implementation Tasks

## GPU Kernel and API

- [x] In `particula/gpu/kernels/coagulation.py`, add the charged-term majorant
  branch to the E5-F1 majorant dispatcher using all unique compact active pairs.
- [x] Pass E5-F2's charge array, prepared total masses, and charged pair
  properties into the shared selector without a mechanism-specific collision
  buffer.
- [x] Register only exact charged-only particle-resolved capability after the
  charged helper, majorant, validation, and merge prerequisites are present.
- [x] Route charged-only candidate rates through the shared acceptance ratio and
  existing disjoint active-rank removal.
- [x] Register canonical Brownian-plus-charged capability and sum both candidate
  rates before one acceptance draw.
- [x] Use one exhaustive additive-rate majorant scan for combined execution and
  retain E5-F1 finite/non-negative device guards.
- [x] Preserve omitted-config Brownian behavior, keyword-only API shape, return
  tuple, output-buffer identity, and persistent RNG reset/reuse semantics.
- [x] Update kernel, entry-point, and charged helper docstrings with the exact
  supported charged model and ownership behavior.

## Tooling / Tests

- [x] Add deterministic charged-majorant probes to
  `particula/gpu/kernels/tests/coagulation_test.py` and compare every active pair
  against independently calculated expected rates, including invalid/zero,
  sparse-list, and per-box cases.
- [x] Add charged-only seeded fresh-run aggregate statistics for same-sign,
  opposite-sign, neutral-limit, and mixed-scale fixtures.
- [x] Assert charged accepted pairs are sorted, distinct, in bounds, initially
  active, and disjoint for each box.
- [x] Assert per-species mass and total charge conservation for charged-only
  calls, including donor-field clearing from E5-F2.
- [x] Add snapshots proving invalid charged capability/model/input requests,
  including non-finite charge, leave particles, output buffers, and persistent
  RNG state unchanged.
- [x] Add Brownian-plus-charged additive-rate and total-majorant probes and prove
  only one RNG/selection pass occurs.
- [x] Add caller-owned buffer identity, persistent RNG, inactive-slot,
  multi-box, capacity, and multi-species total-mass tests for charged-only.
- [x] Parameterize charged evidence through existing Warp CPU/CUDA helpers.
- [x] Keep changed-code coverage at or above 80% without lowering thresholds and
  run focused lint and pytest validation.
