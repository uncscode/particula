# Feature E5-F7: Warp/GPU Translation (Follow-on)

**Parent Epic**: [E5: Non-Isothermal Condensation with Latent Heat](../epics/E5-non-isothermal-condensation.md)
**Status**: Planning
**Priority**: P2
**Owners**: @Gorkowski
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-03-02
**Size**: Medium (3 phases)

## Summary

Translate the Python-native non-isothermal condensation implementation to
NVIDIA Warp (`wp.func` / `wp.kernel`) for GPU acceleration. This is a follow-on
feature that ships after the Python implementation (E5-F1 through E5-F6) is
complete and validated. The GPU translation follows patterns established in
E3-F3 (Warp Integration and GPU Kernels).

## Goals

1. Translate `get_thermal_resistance_factor` and
   `get_mass_transfer_rate_latent_heat` to Warp `wp.func` equivalents
2. Create GPU kernel for non-isothermal condensation step including energy
   tracking
3. Numerical parity tests: GPU vs CPU results to within floating-point tolerance
4. Performance comparison: CPU vs GPU for large particle counts
5. Documentation and examples for GPU usage

## Non-Goals

- Modifying the Python-native implementation
- Adding new physics or features beyond what Python already supports
- Supporting non-NVIDIA GPUs (Warp is CUDA-only)
- Staggered GPU variant

## Dependencies

- **E5-F1 through E5-F5** must be complete (Python implementation is the
  reference)
- **E3-F3** (Warp Integration) should be sufficiently advanced to provide
  GPU kernel patterns
- NVIDIA Warp package must be available in the environment
- CUDA-capable GPU for testing

## Design

### Warp Function Translation

The pure functions from `mass_transfer.py` translate to `wp.func` annotated
functions. These are device-callable helper functions (not full kernels):

```python
@wp.func
def thermal_resistance_factor_wp(
    diffusion_coefficient: float,
    latent_heat: float,
    vapor_pressure_surface: float,
    thermal_conductivity: float,
    temperature: float,
    molar_mass: float,
) -> float:
    """Warp equivalent of get_thermal_resistance_factor."""
    r_specific = GAS_CONSTANT / molar_mass
    return (
        (diffusion_coefficient * latent_heat * vapor_pressure_surface)
        / (thermal_conductivity * temperature)
        * (latent_heat / (r_specific * temperature) - 1.0)
        + r_specific * temperature
    )
```

### GPU Kernel

The condensation step kernel processes all particles in parallel:

```python
@wp.kernel
def condensation_latent_heat_step_kernel(
    # Input arrays (per-particle)
    particle_radius: wp.array(dtype=float),
    particle_mass: wp.array2d(dtype=float),  # (n_particles, n_species)
    concentration: wp.array(dtype=float),
    # ... (temperature, pressure, gas state, strategy params)
    # Output arrays
    mass_change: wp.array2d(dtype=float),
    energy_released: wp.array(dtype=float),
):
    tid = wp.tid()
    # Per-particle non-isothermal condensation computation
    ...
```

## Phase Checklist

- [ ] **E5-F7-P1**: Translate latent heat pure functions to `wp.func` kernels
  with tests
  - Issue: TBD | Size: M (~80 LOC) | Status: Not Started
  - Translate `get_thermal_resistance_factor` and
    `get_mass_transfer_rate_latent_heat` to Warp `wp.func` equivalents
  - Follow patterns established in E3-F3 for Warp integration
  - Add Warp unit tests mirroring Python tests with numerical parity
  - Tests: GPU vs CPU parity for thermal resistance factor and mass transfer
    rate (relative tolerance < 1e-6 for float32, < 1e-12 for float64)

- [ ] **E5-F7-P2**: Translate `CondensationLatentHeat.step()` to Warp kernel
  with tests
  - Issue: TBD | Size: L (~120 LOC) | Status: Not Started
  - Create GPU kernel for non-isothermal condensation step
  - Include energy tracking in kernel output
  - Handle all particle data in GPU arrays (mass, radius, concentration)
  - Parity test: GPU vs CPU results to within floating-point tolerance
  - Performance test (marked `@pytest.mark.slow`): compare GPU vs CPU
    execution time for 1000, 10000, 100000 particles

- [ ] **E5-F7-P3**: Update development documentation for GPU feature
  - Issue: TBD | Size: XS (~30 LOC) | Status: Not Started
  - Update epic and feature docs with GPU completion status
  - Add GPU usage examples to existing notebook
    (`docs/Examples/Dynamics/non_isothermal_condensation_example.py`)
  - Document performance comparison (CPU vs GPU)
  - Update `adw-docs/dev-plans/features/index.md`

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds as shipped (80%).
- **Self-Contained Tests**: GPU parity tests that compare against Python
  reference.
- **Test-First Completion**: Write and pass tests before declaring phases ready
  for review.
- **Numerical Parity**: GPU results must match CPU to within floating-point
  tolerance (< 1e-6 for float32, < 1e-12 for float64).
- **Performance Tests**: Marked with `@pytest.mark.slow` and
  `@pytest.mark.performance` so they are excluded from CI.

## Testing Strategy

### Unit Tests

| Test File | Phase | Coverage Target |
|-----------|-------|----------------|
| TBD (follows E3-F3 test patterns) | P1 | Warp functions |
| TBD (follows E3-F3 test patterns) | P2 | Warp kernels |

### Key Test Cases

1. **Function parity**: `thermal_resistance_factor_wp` matches
   `get_thermal_resistance_factor` for same inputs
2. **Rate parity**: `mass_transfer_rate_latent_heat_wp` matches Python version
3. **Kernel parity**: Full step kernel matches `CondensationLatentHeat.step()`
4. **Energy tracking**: GPU energy output matches CPU sum(dm * L)
5. **Performance**: GPU faster than CPU for >= 10000 particles (marked slow)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| E3-F3 Warp patterns not yet finalized | Feature is explicitly deferred; wait for E3-F3 to stabilize |
| Float32 precision loss on GPU | Test both float32 and float64; document precision trade-offs |
| Warp not available in all environments | Skip GPU tests when Warp not installed (`pytest.importorskip`) |
| Performance regression | Benchmark tests with explicit particle count thresholds |

## Success Criteria

1. Warp `wp.func` functions produce numerically identical results to Python
2. GPU condensation kernel works for particle-resolved distribution
3. GPU faster than CPU for >= 10000 particles
4. All GPU tests pass with Warp installed
5. Tests gracefully skip when Warp is not available
6. Documentation covers GPU usage and performance characteristics

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-02 | Initial feature document created from E5 epic | ADW |
