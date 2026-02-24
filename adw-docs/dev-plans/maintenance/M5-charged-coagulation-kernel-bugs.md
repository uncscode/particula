# Maintenance M5: Charged Coagulation Kernel Bugs

**ID:** M5
**Priority:** P1
**Status:** Shipped
**Last Updated:** 2026-02-24

## Vision

The charged coagulation kernel has two related bugs that produce incorrect
simulation results. These must be investigated, reproduced with pytest, and
fixed to ensure physically correct charged particle coagulation simulations.

## Scope

**Affected Modules/Components:**

- `particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method`
  -- bin-pair iteration and kernel interpolation logic
- `particula.dynamics.coagulation.charged_dimensional_kernel`
  -- system state property calculations
- `particula.dynamics.coagulation.charged_dimensionless_kernel`
  -- dimensional kernel formula
- `particula.particles.properties.diffusive_knudsen_module`
  -- diffusive Knudsen number computation
- `particula.particles.properties.coulomb_enhancement`
  -- Coulomb enhancement factors

**Affected Files/Directories:**

- `particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py`
- `particula/dynamics/coagulation/charged_dimensional_kernel.py`
- `particula/dynamics/coagulation/charged_dimensionless_kernel.py`
- `particula/particles/properties/diffusive_knudsen_module.py`
- `particula/particles/properties/coulomb_enhancement.py`
- `particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py`
- `particula/dynamics/coagulation/tests/`

## Bug Descriptions

### Bug 1: NaN kernel for monodisperse particles

When all particles have identical radii (e.g., `np.full(N, 50e-9)`), the
charged coagulation kernel produces NaN values. The crash occurs at
`particle_resolved_method.py:202`:

```python
tests = int(np.ceil(kernel_values.item() * time_step * events / volume))
# ValueError: cannot convert float NaN to integer
```

The NaN propagates through the Coulomb enhancement chain. For same-sign
self-interaction (e.g., charge -6 with itself), `get_coulomb_enhancement_ratio`
computes a large negative potential ratio (clipped to `ratio_lower_limit=-200`).
Then `get_coulomb_kinetic_limit` returns `exp(-200) ≈ 0`. In the diffusive
Knudsen number (`diffusive_knudsen_module.py:124`), the denominator
`sum_of_radii * continuum_enhance / kinetic_enhance` divides by this near-zero
`kinetic_enhance`, producing inf or NaN in the Knudsen number. This cascades
into the dimensional kernel formula:

```python
dimensionless_kernel * reduced_friction_factor * sum_of_radii**3
    * coulomb_kinetic_limit**2 / (reduced_mass * coulomb_continuum_limit)
```

Note: `reduced_mass` is NOT the problem -- `get_reduced_self_broadcast`
explicitly guards against zero denominators. The root cause is the
`kinetic_enhance → 0` path in the Coulomb enhancement chain.

**Reproduction:** Create two populations with identical radii (e.g., 50nm
calcite at charge -6, 50nm ions at charge +1). Run one coagulation step.

**Workaround:** Sample radii from a narrow lognormal distribution (GSD=1.05)
instead of `np.full()`.

### Bug 2: Spurious same-sign coagulation driven by small ions

When 5nm ions (charge +1) are present alongside 50nm calcite (charge -6),
the coagulation step produces ~2,600 calcite-calcite mergers in a single
0.034s time step. However, when ions are removed (calcite-only), zero
coagulation events occur -- as expected given the strong Coulomb repulsion
between -6 charged particles.

The small ions create a different kernel radius bin. When the particle-resolved
method iterates over bin pairs, the small-ion bin paired with the calcite bin
produces a large kernel value. This inflated kernel value appears to affect
the coagulation probability calculations for calcite-calcite bin pairs,
causing same-sign particles to coagulate when the Coulomb barrier should
forbid it.

**Reproduction:**
1. Run 10k calcite (50nm, charge -6) + 6 ions (5nm, charge +1) in scaled
   volume -- observe ~2,600 calcite-calcite mergers
2. Run 10k calcite (50nm, charge -6) only, no ions -- observe zero mergers
3. The difference proves the ions cause spurious calcite-calcite coagulation

**Evidence from diagnostic runs:**

| Scenario | Calcite mergers | Ion-calcite mergers |
|----------|----------------|---------------------|
| Calcite + ions (5nm) | ~2,594 | 5-6 |
| Calcite only | 0 | N/A |
| Calcite + ions (50nm, lognormal) | 0 | 0 |

## Dependencies

- None (self-contained bug fix)

## Phases

Each phase writes tests AND fixes the bug together so that every commit
leaves the test suite green. No failing tests are committed to the repo.

### Phase 1: Fix NaN kernel for monodisperse particles (M5-P1)

**Scope:** Investigate, reproduce, and fix Bug 1 (NaN kernel) in one phase.
Write passing tests that verify the fix.

**Size:** M (~100-150 LOC)

**Tasks:**

- [ ] Create test file
      `particula/dynamics/coagulation/tests/charged_kernel_bugs_test.py`
- [ ] Confirm the NaN chain: `coulomb_potential_ratio → -200` →
      `kinetic_enhance = exp(-200) ≈ 0` → `Kn_d = inf` → `kernel = NaN`
- [ ] Check `get_diffusive_knudsen_number()` -- does the denominator
      `sum_of_radii * continuum_enhance / kinetic_enhance` go to inf
      for strongly repulsive same-sign particles?
- [ ] Check `get_coulomb_enhancement_ratio()` -- does the self-interaction
      diagonal (particle with itself) produce extreme negative ratios?
- [ ] Implement fix: guard against near-zero `kinetic_enhance` in the
      diffusive Knudsen number calculation, OR clamp inf/NaN kernel values
      to zero (self-interaction of same-sign particles should produce zero
      coagulation rate, which is physically correct)
- [ ] Write passing test: Monodisperse NaN -- create identical-radius
      particles with opposite charges, assert kernel computation does not
      produce NaN, assert coagulation step completes without error
- [ ] Write passing test: Same-sign repulsion -- create calcite-only
      population (charge -6), run coagulation step, assert zero coagulation
      events (no mergers between same-sign repulsive particles)
- [ ] Write passing test: Full kernel matrix contains no NaN/Inf for any
      particle population tested
- [ ] Verify no regression in existing coagulation tests

**Acceptance Criteria:**

- [ ] Root cause documented with specific function and line number
- [ ] Monodisperse NaN test passes (kernel is finite, step completes)
- [ ] Same-sign repulsion test passes (zero mergers)
- [ ] Full kernel matrix contains no NaN/Inf
- [ ] All existing coagulation tests still pass
- [ ] All committed tests are green

### Phase 2: Fix spurious same-sign coagulation (M5-P2)

**Scope:** Investigate and fix Bug 2 (spurious calcite-calcite mergers when
small ions are present). Write passing tests that verify the fix.

**Size:** M (~100-150 LOC)

**Tasks:**

- [ ] Investigate the charge-blind interpolation: `_interpolate_kernel`
      (`particle_resolved_method.py:267-308`) creates a
      `RegularGridInterpolator` indexed by radius only. When small ions
      (5nm) create a different radius bin, the kernel matrix encodes
      charge-dependent rates, but the interpolator maps radius pairs
      without considering which charge state produced each kernel entry.
      Determine if this causes wrong kernel lookups for calcite-calcite
      pairs when ions are present.
- [ ] Determine if the issue is in:
  - (a) Kernel interpolation -- does `interp_kernel` extrapolate incorrectly
    when particles are outside the kernel radius range? (`fill_value=None`
    uses nearest-neighbor, which may return an ion-calcite kernel value
    for a calcite-calcite radius pair)
  - (b) Bin assignment -- are small ions and large calcite being placed in
    the same bin, causing wrong kernel lookups?
  - (c) Probability calculation -- is the `events / (tests * volume)` ratio
    inflated by the ion bin contribution?
  - (d) Coulomb potential sign -- is the repulsive potential being ignored
    or incorrectly applied in the hard-sphere kernel path?
  - (e) NaN contamination from Bug 1 -- if the kernel matrix has NaN/inf
    entries due to the kinetic_enhance→0 issue, does the interpolator
    propagate these into non-NaN bins?
- [ ] Implement fix for the identified root cause
- [ ] Write passing test: Spurious coagulation -- create calcite
      (50nm, charge -6) + ions (5nm, charge +1), run coagulation step,
      assert calcite-calcite mergers are zero or negligible (only
      ion-calcite mergers should occur)
- [ ] Write passing test: Conservation -- for any coagulation step, assert
      total mass and total charge are conserved
- [ ] Verify: calcite-only still produces zero mergers (Phase 1 test)
- [ ] Verify: calcite + ions produces only ion-calcite mergers, not
      calcite-calcite mergers

**Acceptance Criteria:**

- [ ] Calcite-calcite coagulation rate is zero (or negligible) when all
      calcite particles carry same-sign charge
- [ ] Ion-calcite coagulation still works correctly
- [ ] Mass and charge conservation test passes
- [ ] All existing coagulation tests still pass
- [ ] All committed tests are green

### Phase 3: Documentation and cleanup (M5-P3)

**Scope:** Update dev docs, clean up duplicate code, ensure all tests pass.

**Size:** S

**Tasks:**

- [ ] Clean up duplicate code in `charged_dimensional_kernel.py`:
  - Remove 4 of 5 identical `np.atleast_1d(dimensionless_kernel_raw)` calls
    in `get_hard_sphere_kernel_via_system_state` (lines 190-202)
  - Remove unreachable duplicate `return get_dimensional_kernel(...)` in
    `get_coulomb_kernel_dyachkov2007_via_system_state` (lines 278-284)
- [ ] Remove any diagnostic logging added in Phases 1-2
- [ ] Update this maintenance document status to Shipped
- [ ] Run full test suite (`pytest --cov=particula`)
- [ ] Ensure no coverage regression

## Technical Approach

### Key files to investigate

1. **`coulomb_enhancement.py:83-97`** -- The Coulomb potential ratio
   computation. Self-interaction (same-sign particle with itself) produces
   `phi_E = -q^2 * e^2 / (4*pi*eps0 * 2r * kT)` → large negative,
   clipped to -200. **This is the root entry point of the NaN chain.**

2. **`coulomb_enhancement.py:136-140`** -- `get_coulomb_kinetic_limit`:
   for `phi_E = -200`, returns `exp(-200) ≈ 0`. This near-zero value
   causes division-by-zero downstream.

3. **`diffusive_knudsen_module.py:119-126`** -- The Knudsen number formula
   denominator is `sum_of_radii * continuum_enhance / kinetic_enhance`.
   When `kinetic_enhance ≈ 0`, division produces inf. **This is where
   the NaN first materializes.**

4. **`charged_dimensionless_kernel.py:50-56`** -- The dimensional kernel
   formula `dimensionless_kernel * reduced_friction_factor * sum_of_radii^3
   * coulomb_kinetic_limit^2 / (reduced_mass * coulomb_continuum_limit)`.
   Note: `reduced_mass` is fine (guarded by `get_reduced_self_broadcast`);
   the inf Knudsen number from step 3 is the actual problem.

5. **`particle_resolved_method.py:188-208`** -- The bin-pair loop where
   kernel values are sampled and test counts are computed. Line 202 is
   where NaN crashes. The `interp_kernel` call at line 191 may return NaN
   for degenerate radius pairs.

6. **`particle_resolved_method.py:267-308`** -- `_interpolate_kernel`
   creates a `RegularGridInterpolator` indexed by radius only, with
   `fill_value=None` (nearest-neighbor extrapolation). This is
   charge-blind and may return wrong kernel values when particles fall
   outside the `kernel_radius` range.

### Pre-existing code issues (discovered during review)

7. **`charged_dimensional_kernel.py:190-202`** --
   `get_hard_sphere_kernel_via_system_state` contains
   `np.atleast_1d(dimensionless_kernel_raw)` repeated **5 times**
   identically. Only one call is needed.

8. **`charged_dimensional_kernel.py:271-284`** --
   `get_coulomb_kernel_dyachkov2007_via_system_state` has a **duplicate
   return statement** (`return get_dimensional_kernel(...)` appears twice).
   The second is unreachable dead code.

### Likely root causes

**Bug 1 (NaN):** The Coulomb enhancement chain produces degenerate values
for same-sign self-interaction. Specifically:

1. `get_coulomb_enhancement_ratio` for particle i with itself (same sign):
   `phi_E = -q^2 * e^2 / (4*pi*eps0 * 2r * kT)` → large negative (clipped
   to -200).
2. `get_coulomb_kinetic_limit(phi=-200)` → `exp(-200) ≈ 0`.
3. In `get_diffusive_knudsen_number`, the denominator becomes
   `sum_of_radii * continuum_enhance / kinetic_enhance` where
   `kinetic_enhance ≈ 0` → division by near-zero → inf Knudsen number.
4. The inf Knudsen number cascades through `get_dimensional_kernel()` into
   NaN in the final kernel matrix.

Note: `reduced_mass = m*m/(m+m) = m/2` is fine -- `get_reduced_self_broadcast`
has an explicit zero-denominator guard. The reduced mass is NOT the issue.

**Bug 2 (spurious coagulation):** The `_interpolate_kernel` in
`particle_resolved_method.py` creates a `RegularGridInterpolator` that
indexes the kernel matrix by **radius only** -- it does not carry charge
information. When the kernel matrix is computed for a mixed population
(ions + calcite), the interpolation maps radius pairs to kernel values,
but the kernel matrix encodes different charge-dependent rates for
different radius bins. When particles fall outside the `kernel_radius`
range, the interpolator uses nearest-neighbor extrapolation
(`fill_value=None`), which can return the kernel value from an adjacent
bin with very different charge characteristics.

Additionally, the `HardSphereKernelStrategy.dimensionless()` ignores the
Coulomb potential ratio -- it only uses `diffusive_knudsen`. The
dimensional conversion in `get_dimensional_kernel()` applies Coulomb
enhancement factors `Gamma_kin^2 / Gamma_cont`, but for same-sign
repulsive particles this ratio may not suppress the kernel sufficiently,
especially when the Knudsen number is already corrupted by the
kinetic_enhance → 0 issue from Bug 1.

## Testing Strategy

### Unit Tests

- `particula/dynamics/coagulation/tests/charged_kernel_bugs_test.py`
  - Monodisperse NaN reproduction
  - Same-sign repulsion verification
  - Spurious cross-bin coagulation detection
  - Mass and charge conservation checks

### Integration Tests

- Full charged coagulation step with mixed populations
- Verify physical correctness: opposite-sign attracts, same-sign repels

## Success Criteria

- [ ] Monodisperse particles (identical radii) do not produce NaN
- [ ] Full kernel matrix contains no NaN/Inf for any particle population
- [ ] Same-sign charged particles do not coagulate
- [ ] Ion-calcite (opposite sign) coagulation works correctly
- [ ] Mass and charge are conserved in all scenarios
- [ ] All existing coagulation tests still pass
- [ ] Duplicate code in `charged_dimensional_kernel.py` is cleaned up
- [ ] Coverage >= 80% on changed code

## References

- `private_dev/charge_simV2_test.py` -- diagnostic script that discovered
  both bugs
- `particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py`
- Gopalakrishnan & Hogan (2012), Phys. Rev. E 85(2) -- Coulomb-influenced
  collisions theory

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-23 | Initial maintenance plan created | ADW Workflow |
| 2026-02-23 | Corrected root cause analysis (kinetic_enhance→0, not reduced_mass→0); upgraded P2 to size M; added kernel NaN/Inf success criterion; strengthened P3 investigation on charge-blind interpolation; added duplicate code cleanup to P4 | ADW Review |
| 2026-02-23 | Restructured phases: merged test-writing into fix phases so no failing tests are committed. P1 = fix NaN + tests, P2 = fix spurious coagulation + tests, P3 = cleanup. Removed standalone reproducer-tests phase. | Manual |
| 2026-02-24 | Shipped P3 cleanup (removed dead code, verified no diagnostic logging). | ADW Workflow |
