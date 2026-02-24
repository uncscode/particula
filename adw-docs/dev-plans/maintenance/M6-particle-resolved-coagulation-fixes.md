# Maintenance M6: Particle-Resolved Coagulation Fixes

**ID:** M6
**Priority:** P1
**Status:** Planning
**Last Updated:** 2026-02-24

## Vision

The particle-resolved coagulation path has two bugs that produce physically
incorrect results for charged particle simulations:

1. **Duplicate-index mass/charge loss**: When a large particle absorbs
   multiple smaller particles in a single step, NumPy's buffered fancy-index
   `+=` silently drops all but the last addition. Both mass and charge are
   lost.

2. **Charge-blind kernel interpolation**: The radius-only kernel interpolator
   bleeds cross-charge kernel values into wrong particle pairs, causing
   spurious same-sign coagulation when mixed-charge populations are present.

Both bugs were discovered and characterized using diagnostic tests in
`private_dev/`. This plan fixes both bugs, adds regression tests, moves the
diagnostics into the repo, and adds an opt-in direct kernel evaluation mode
for particle-resolved coagulation.

## Scope

**Affected Modules/Components:**

- `particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method`
  -- `get_particle_resolved_update_step()` and `collide_pairs` delegation
- `particula.particles.distribution_strategies.particle_resolved_speciated_mass`
  -- `collide_pairs()` vectorized mass/charge merging
- `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc`
  -- `step()` method for particle_resolved path, new `use_direct_kernel` flag
- `particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy`
  -- inherits the new flag

**Affected Files:**

- `particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py`
- `particula/particles/distribution_strategies/particle_resolved_speciated_mass.py`
- `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py`
- `particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py`
- `particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py`
- Tests in `particula/dynamics/coagulation/tests/`
- Tests in `particula/particles/distribution_strategies/tests/`
- Integration tests in `particula/integration_tests/`

## Bug Details

### Bug 1: Duplicate-Index Mass/Charge Loss

**Root Cause:** NumPy fancy indexing with `+=` is buffered. When
`large_index` contains duplicates (e.g., particle X absorbs both A and B),
only the last assignment takes effect:

```python
# If large_index = [5, 5] and small_index = [2, 3]:
distribution[large_index] += distribution[small_index]
# Result: distribution[5] gets only distribution[3], NOT both 2 and 3
```

**Affected locations:**

1. `particle_resolved_speciated_mass.py:223-227` -- mass merge in
   `collide_pairs()`
2. `particle_resolved_speciated_mass.py:237-238` -- charge merge in
   `collide_pairs()`
3. `particle_resolved_method.py:41-57` -- radius update in
   `get_particle_resolved_update_step()`

**Why it wasn't caught:** The existing `test_mass_charge_conservation`
(line 312 of `particle_resolved_speciated_mass_test.py`) computes charge
conservation manually with a Python for-loop, which correctly handles
duplicates. The production code uses vectorized `+=`, which does not.

**Fix:** Replace `a[idx] += b[idx]` with `np.add.at(a, idx, b[idx])`.
`np.add.at` is unbuffered and correctly accumulates for duplicate indices.

### Bug 2: Charge-Blind Kernel Interpolation

**Root Cause:** The kernel matrix is computed on binned representatives
where each bin has a median charge. The `RegularGridInterpolator` maps
`(radius_small, radius_large) -> kernel_value`, discarding charge. When
ions (50nm, charge +6) and calcite (100nm, charge -6) are present:

- The ion-calcite kernel entry is ~1e-13 (attractive, 22x Brownian)
- The calcite-calcite entry is ~3e-24 (repulsive, ~0)
- The interpolator linearly blends these in the radius transition zone,
  giving calcite-calcite pairs artificially high kernel values

**Evidence from kernel interpolation diagnostic (Scenario 6):**

| Radius (nm) | K interpolated | K(+6,-6) direct | K(-6,-6) direct |
|-------------|---------------|-----------------|-----------------|
| 50          | 1.01e-13      | 1.01e-13        | 2.09e-26        |
| 60          | 6.09e-14      | 8.48e-14        | 6.56e-26        |
| 70          | 2.03e-14      | 7.46e-14        | 1.88e-25        |
| 80          | 8.02e-25      | 6.80e-14        | 4.95e-25        |

The interpolator gives 80nm particles (calcite) a kernel of 8e-25 instead
of the correct 5e-25, but worse, particles near the 60-70nm boundary get
kernel values 10+ orders of magnitude too high.

**Fix:** Add opt-in `use_direct_kernel=True` flag on the coagulation
strategy. When enabled, compute the kernel directly for each sampled
particle pair during the particle-resolved step, bypassing the binned
kernel matrix and interpolator entirely.

## Dependencies

- M5 (Charged Coagulation Kernel Bugs) -- Shipped. M6 addresses remaining
  bugs discovered after M5 shipped.
- No external dependencies.

## Phase Checklist

### Phase 1: Fix duplicate-index mass/charge loss (M6-P1)

**Scope:** Fix the NumPy buffered fancy-indexing bug in both
`collide_pairs()` and `get_particle_resolved_update_step()`. Write
regression tests that fail before the fix and pass after.

**Size:** S (~60-80 LOC)

**Files to modify:**

- `particula/particles/distribution_strategies/particle_resolved_speciated_mass.py`
  -- `collide_pairs()` method
- `particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py`
  -- `get_particle_resolved_update_step()` function

**Tasks:**

- [ ] Fix `collide_pairs()`: replace `distribution[large_index] +=
  distribution[small_index]` with `np.add.at(distribution, large_index,
  distribution[small_index])` for both 1D and 2D cases
- [ ] Fix `collide_pairs()`: replace `charge[large_index] +=
  charge[small_index]` with `np.add.at(charge, large_index,
  charge[small_index])`
- [ ] Fix `get_particle_resolved_update_step()`: handle duplicate
  `large_index` in the radius cubed summation. Pre-compute `new_radii`
  correctly using `np.add.at` on the cubed-radius accumulation
- [ ] Add test: `test_collide_pairs_duplicate_large_index` -- two small
  particles merge into the same large particle, verify mass is sum of
  all three, charge is sum of all three
- [ ] Add test: `test_collide_pairs_duplicate_large_mass_conservation` --
  generate random pairs with known duplicates, assert total mass before
  equals total mass after
- [ ] Add test: `test_collide_pairs_duplicate_large_charge_conservation` --
  same as above for charge
- [ ] Add test: `test_update_step_duplicate_large_volume_conservation` --
  verify `get_particle_resolved_update_step` conserves volume when
  `large_index` has duplicates
- [ ] Run existing tests, verify no regressions

**Acceptance Criteria:**

- [ ] `np.add.at` used in all three locations (mass, charge, radius)
- [ ] All new tests pass
- [ ] All existing `collide_pairs` tests pass
- [ ] Total mass and charge conserved for duplicate-index scenarios
- [ ] All committed tests are green

### Phase 2: Add opt-in direct kernel evaluation (M6-P2)

**Scope:** Add a `use_direct_kernel` flag (default `False`) to the
coagulation strategy ABC. When `True`, the particle-resolved `step()`
method computes the kernel directly for each sampled particle pair instead
of using the binned kernel matrix + interpolator.

**Size:** M (~100-150 LOC)

**Files to modify:**

- `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py`
  -- add `use_direct_kernel` parameter to `__init__`, modify `step()`
- `particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py`
  -- add `get_particle_resolved_coagulation_step_direct()` or modify
  existing function to accept a kernel callable
- `particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py`
  -- add `set_use_direct_kernel()` builder method
- `particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py`
  -- wire through the flag

**Tasks:**

- [ ] Add `use_direct_kernel: bool = False` parameter to
  `CoagulationStrategyABC.__init__()`
- [ ] Add `set_use_direct_kernel(value: bool)` to the builder mixin
- [ ] Wire through `ChargedCoagulationBuilder`
- [ ] In `CoagulationStrategyABC.step()`, when `use_direct_kernel=True`
  and `distribution_type == "particle_resolved"`:
  - Still bin particles and select pairs using the same binning logic
  - Instead of interpolating kernel from the matrix, call
    `self.kernel()` directly for each sampled (small, large) pair
  - This requires building a mini 2-particle representation for each
    pair, or passing a kernel callable to the step function
- [ ] Alternative approach: pass a `kernel_func(radius_pairs) -> values`
  callable to `get_particle_resolved_coagulation_step()` that replaces
  the interpolator. When `use_direct_kernel=True`, this callable
  evaluates the full charged kernel for each pair. When `False`, it
  uses the existing interpolator. This minimizes changes to the step
  function signature.
- [ ] Add test: `test_direct_kernel_same_sign_no_spurious_mergers` --
  calcite(-6) + ions(+6), `use_direct_kernel=True`, verify
  calcite-calcite mergers are zero
- [ ] Add test: `test_direct_kernel_opposite_sign_attracts` -- verify
  ion-calcite mergers still occur with direct kernel
- [ ] Add test: `test_direct_kernel_flag_default_false` -- verify
  default behavior is unchanged
- [ ] Run full test suite

**Acceptance Criteria:**

- [ ] `use_direct_kernel=True` produces correct charge-aware kernel values
  for all particle pairs
- [ ] Calcite-calcite mergers are zero (or negligible) when all calcite
  has same-sign charge, even with ions present
- [ ] Ion-calcite coagulation still works correctly
- [ ] Default behavior (`use_direct_kernel=False`) unchanged
- [ ] Builder supports setting the flag
- [ ] All committed tests are green

### Phase 3: Move diagnostics into repo and add integration tests (M6-P3)

**Scope:** Move the `private_dev/` diagnostic tests into the repo as
proper integration tests. Add a charged coagulation comparison integration
test that validates physical behavior end-to-end.

**Size:** S (~60-80 LOC)

**Files to create/modify:**

- `particula/integration_tests/charged_coagulation_comparison_test.py`
  -- adapted from `private_dev/charge_coagulation_comparison_test.py`
- `particula/dynamics/coagulation/tests/kernel_interpolation_diagnostic_test.py`
  -- adapted from `private_dev/kernel_interpolation_diagnostic_test.py`

**Tasks:**

- [ ] Move `charge_coagulation_comparison_test.py` logic into
  `particula/integration_tests/charged_coagulation_comparison_test.py`
  as a proper pytest test with assertions (not just print output)
- [ ] Move `kernel_interpolation_diagnostic_test.py` scenarios 4-6 into
  `particula/dynamics/coagulation/tests/kernel_interpolation_diagnostic_test.py`
  with assertions validating kernel values
- [ ] Add assertions for physical expectations:
  - Same-sign ions: zero ion-calcite mergers, zero calcite-calcite mergers
  - Neutral ions: some ion-calcite mergers, some calcite-calcite mergers
    (Brownian baseline)
  - Opposite-sign ions (with `use_direct_kernel=True`): many ion-calcite
    mergers, zero calcite-calcite mergers, charge conserved
- [ ] Add mass and charge conservation assertions to all integration tests
- [ ] Run full test suite

**Acceptance Criteria:**

- [ ] Integration tests pass with assertions (not just print diagnostics)
- [ ] Charge conservation verified in all scenarios
- [ ] Mass conservation verified in all scenarios
- [ ] Physical behavior matches expectations for all three charge cases
- [ ] All committed tests are green

### Phase 4: Update development documentation (M6-P4)

**Scope:** Update dev docs, mark M6 shipped, clean up.

**Size:** XS

**Tasks:**

- [ ] Update `adw-docs/dev-plans/maintenance/index.md` with final status
- [ ] Update `adw-docs/dev-plans/README.md` if needed
- [ ] Add completion notes to this document
- [ ] Run full test suite (`pytest --cov=particula`)
- [ ] Ensure no coverage regression

## Technical Approach

### Phase 1: `np.add.at` Fix

The fix is straightforward. Replace all buffered fancy-index `+=` with
unbuffered `np.add.at`:

**Before (broken):**
```python
distribution[large_index] += distribution[small_index]
```

**After (correct):**
```python
small_mass = distribution[small_index].copy()  # snapshot before zeroing
distribution[small_index] = 0
np.add.at(distribution, large_index, small_mass)
```

Note: we must snapshot `small_index` values before zeroing, in case a
particle appears in both `small_index` and `large_index` (chain merges).
The same pattern applies to charge and radius updates.

For `get_particle_resolved_update_step`, the radius update needs
restructuring since it computes `new_radii = cbrt(r_small^3 + r_large^3)`
per pair. With duplicate `large_index`, we need to accumulate volumes:

```python
# Accumulate small volumes onto large particles
volume_addition = np.zeros_like(particle_radius)
np.add.at(volume_addition, large_index,
          particle_radius[small_index] ** 3)
particle_radius[small_index] = 0
particle_radius[large_index] = np.cbrt(
    particle_radius[large_index] ** 3 + volume_addition[large_index]
)
```

### Phase 2: Direct Kernel Callable

The cleanest approach is to pass a kernel callable to the step function:

```python
def get_particle_resolved_coagulation_step(
    particle_radius, kernel, kernel_radius, volume, time_step,
    random_generator,
    kernel_func=None,  # New: optional direct kernel callable
):
```

When `kernel_func` is not None, steps 5 and 9 call `kernel_func` instead
of `interp_kernel`:

```python
# Step 5: kernel for bin pair bounds
if kernel_func is not None:
    kernel_values = kernel_func(
        np.array([small_sample]), np.array([large_sample])
    )
else:
    kernel_values = interp_kernel(...)

# Step 9: kernel for sampled pairs
if kernel_func is not None:
    kernel_value = kernel_func(
        particle_radius[small_index], particle_radius[large_index]
    )
else:
    kernel_value = interp_kernel(...)
```

The `CoagulationStrategyABC.step()` method builds the callable from
`self.kernel()` when `use_direct_kernel=True`:

```python
if self.use_direct_kernel:
    def kernel_func(r_small, r_large):
        # Build pairwise kernel for actual particle pairs
        # using full charged kernel computation
        ...
    loss_gain_indices = step_func(..., kernel_func=kernel_func)
else:
    loss_gain_indices = step_func(...)  # existing interpolation path
```

## Testing Strategy

### Unit Tests

- `particula/particles/distribution_strategies/tests/particle_resolved_speciated_mass_test.py`
  -- new duplicate-index tests for `collide_pairs()`
- `particula/dynamics/coagulation/tests/particle_resolved_method_test.py`
  -- new duplicate-index test for `get_particle_resolved_update_step()`
- `particula/dynamics/coagulation/tests/kernel_interpolation_diagnostic_test.py`
  -- kernel interpolation scenarios matching coag comparison setup

### Integration Tests

- `particula/integration_tests/charged_coagulation_comparison_test.py`
  -- end-to-end charged coagulation with three ion charge scenarios

### Regression Tests

- All existing tests in `charged_kernel_bugs_test.py` must continue to pass
- All existing `collide_pairs` tests must continue to pass

## Success Criteria

- [ ] Mass conserved in all coagulation scenarios (including duplicate
  large indices)
- [ ] Charge conserved in all coagulation scenarios
- [ ] `use_direct_kernel=True` eliminates spurious same-sign mergers
- [ ] Default behavior (`use_direct_kernel=False`) unchanged
- [ ] Integration tests validate physical correctness for three charge cases
- [ ] All existing tests pass
- [ ] Coverage >= 80% on changed code

## References

- `private_dev/charge_coagulation_comparison_test.py` -- diagnostic that
  revealed charge non-conservation and spurious mergers
- `private_dev/kernel_interpolation_diagnostic_test.py` -- diagnostic that
  characterized the charge-blind interpolation problem
- M5 (Charged Coagulation Kernel Bugs) -- predecessor plan, shipped
- NumPy documentation on `np.add.at` vs fancy-index `+=`:
  https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-24 | Initial maintenance plan created from investigation | ADW |
