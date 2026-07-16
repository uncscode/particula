# Implementation Tasks

## GPU Kernel and API

- [ ] In `particula/gpu/kernels/coagulation.py`, add the charged-term majorant
  branch to the E5-F1 majorant dispatcher using all unique active pairs.
- [ ] Pass E5-F2's charge array and charged pair properties into the shared
  selector without allocating a mechanism-specific collision buffer.
- [ ] Register charged-only capability only after the charged helper, majorant,
  validation, and merge prerequisites are present.
- [ ] Route charged-only candidate rates through the shared acceptance ratio and
  existing disjoint active-rank removal.
- [ ] Register canonical Brownian-plus-charged capability and sum both candidate
  rates before one acceptance draw.
- [ ] Sum the Brownian and charged term bounds for combined execution and retain
  E5-F1 finite/non-negative device guards.
- [ ] Preserve omitted-config Brownian behavior, keyword-only API shape, return
  tuple, output-buffer identity, and persistent RNG reset/reuse semantics.
- [ ] Update kernel and entry-point docstrings with the exact supported charged
  model, mechanism combinations, and ownership behavior.

## Tooling / Tests

- [ ] Add deterministic charged-majorant probes to
  `particula/gpu/kernels/tests/coagulation_test.py` and compare every active pair
  against independently calculated expected rates.
- [ ] Add charged-only seeded fresh-run statistics for same-sign, opposite-sign,
  neutral-limit, and mixed-scale fixtures with declared sigma bounds.
- [ ] Assert accepted pairs are sorted, distinct, in bounds, initially active,
  and disjoint for each box.
- [ ] Assert species mass and total charge separately before/after charged-only
  and combined calls, including donor-field clearing from E5-F2.
- [ ] Add snapshots proving invalid capability/model/input requests leave
  particles, output buffers, and persistent RNG state unchanged.
- [ ] Add Brownian-plus-charged additive-rate and total-majorant probes and prove
  only one RNG/selection pass occurs.
- [ ] Add canonical-order equivalence, legacy Brownian regression, caller-owned
  buffer identity, persistent RNG, inactive-slot, and multi-box tests.
- [ ] Run Warp CPU focused tests when Warp is installed; parameterize optional
  CUDA evidence through existing helpers so unavailable CUDA skips cleanly.
- [ ] Keep changed-code coverage at or above 80% without lowering repository
  thresholds; run Ruff and focused `pytest` commands before phase completion.
