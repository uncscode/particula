# Epic E4: Probabilistic Particle-Resolved Representation

**Status**: Planning
**Priority**: P1
**Owners**: TBD
**Start Date**: TBD
**Target Date**: TBD
**Last Updated**: 2026-01-19
**Size**: L (9 features, ~25 phases)

## Vision

Add a new particle representation that combines the accuracy of particle-resolved
methods with the computational efficiency of super-droplet approaches. Each
computational particle represents a population of physical particles with masses
uniformly distributed within a configurable window (default ±5% of mean, i.e.,
10% total width). This enables simulating millions of physical particles with
thousands of computational particles while preserving distribution information
that pure super-droplet methods lose.

**Key advantages over existing methods:**
- **vs ParticleResolvedSpeciatedMass**: 100-1000x fewer computational particles
  needed for equivalent physical particle counts
- **vs Super-droplet**: Preserves size distribution information within each
  computational particle; more accurate coagulation through sampling
- **vs Binned methods**: Maintains particle identity for tracking mixing state
  and composition evolution

Primary applications: cloud microphysics simulations and aerosol coagulation
where computational cost currently limits particle counts.

## Scope

### In Scope

1. **Core data structure**: `ProbabilisticParticleResolved` distribution strategy
   with per-particle, per-species mass bounds (lower, upper) as 2D arrays and
   concentration as multiplicity
2. **Distribution shape abstraction**: `MassDistributionShape` strategy interface
   with `UniformMassDistribution` implementation (correlated sampling across species)
3. **Extended ParticleRepresentation**: Optional `mass_lower_bound` and
   `mass_upper_bound` 2D arrays (particles × species), `get_probabilistic_radius()` method
4. **Probabilistic condensation**: Mass growth with absolute bound shifts per
   species (captures condensational narrowing of relative distribution width)
5. **Probabilistic coagulation**: Sampled collision products with merge/keep
   decision logic based on distribution overlap
6. **Split/merge maintenance**: Keep computational particle count within 75-100%
   of target via threshold-triggered or manual rebalancing
7. **Builder/factory integration**: Full builder pattern support with default
   relative width and explicit bound override options
8. **Representation conversion**: Convert to/from `ParticleResolvedSpeciatedMass`
9. **Documentation and examples**: Docstrings, Jupyter notebooks, theory docs

### Out of Scope

- GPU/Warp acceleration (see E3)
- Independent (uncorrelated) species variance (using correlated sampling)
- Non-uniform distribution shapes beyond uniform (future extension)
- Charge distribution (single charge per computational particle)
- Automatic variance evolution policies beyond proportional scaling

## Dependencies

- None blocking; this is a new representation that parallels existing ones
- Optional integration with E3 (Data Representation Refactor) for GPU support
  in future phases

## Features

| ID | Name | Status | Phases |
|----|------|--------|--------|
| E4-F1 | [Core Strategy & Data Structure](../features/E4-F1-core-strategy-data.md) | Planning | 4 |
| E4-F2 | [Distribution Shape Interface](../features/E4-F2-distribution-shape-interface.md) | Planning | 2 |
| E4-F3 | [Extended ParticleRepresentation](../features/E4-F3-extended-representation.md) | Planning | 3 |
| E4-F4 | [Probabilistic Condensation](../features/E4-F4-probabilistic-condensation.md) | Planning | 3 |
| E4-F5 | [Probabilistic Coagulation](../features/E4-F5-probabilistic-coagulation.md) | Planning | 5 |
| E4-F6 | [Split/Merge Maintenance](../features/E4-F6-split-merge-maintenance.md) | Planning | 3 |
| E4-F7 | [Builder/Factory Integration](../features/E4-F7-builder-factory.md) | Planning | 3 |
| E4-F8 | [Representation Conversion](../features/E4-F8-representation-conversion.md) | Planning | 2 |
| E4-F9 | [Documentation & Examples](../features/E4-F9-documentation-examples.md) | Planning | 3 |

## Phase Checklist (Epic-Level Milestones)

- [ ] **E4-M1**: Core infrastructure complete (F1, F2, F3)
  - Strategy class with bounds storage
  - Distribution shape interface
  - Extended ParticleRepresentation
  - All unit tests passing

- [ ] **E4-M2**: Dynamics integration complete (F4, F5, F6)
  - Probabilistic condensation working
  - Probabilistic coagulation with sampling
  - Split/merge maintenance operational
  - Integration tests passing

- [ ] **E4-M3**: Production-ready (F7, F8, F9)
  - Builder/factory support
  - Conversion to/from existing representations
  - Documentation and examples complete
  - Performance benchmarks documented

## Technical Design

### Data Model

```python
# Per computational particle i, per species s:
distribution[i, s]       # mean mass per species (kg)
mass_lower_bound[i, s]   # lower mass bound per species (kg) - 2D array
mass_upper_bound[i, s]   # upper mass bound per species (kg) - 2D array
concentration[i]         # multiplicity (# physical particles)
charge[i]                # single charge value (elementary charges)

# Initial constraint: relative_width[s] = (upper[s] - lower[s]) / mean[s] ≈ 0.10 (configurable)
# Sampling is CORRELATED: all species use same scale factor to preserve mass fractions
# Total bounds derived: total_lower = sum(lower[s]), total_upper = sum(upper[s])
```

### Correlated Sampling

When sampling particle masses, all species scale together:
```python
def sample_mass(self, particle_idx: int, rng: np.random.Generator) -> NDArray:
    """Sample mass for all species using correlated scaling."""
    scale = rng.uniform(0, 1)  # single scale factor for all species
    lower = self.mass_lower_bound[particle_idx, :]
    upper = self.mass_upper_bound[particle_idx, :]
    return lower + scale * (upper - lower)
```

This preserves mass fractions at any sampled point while allowing per-species
variance tracking.

### Key Algorithms

**Condensation (absolute shift with condensational narrowing)**:
```python
# Only the condensing species s bounds shift by delta_mass
# This naturally captures condensational narrowing of relative width

def update_bounds_condensation(lower, upper, mean, species_idx, delta_mass):
    """Update bounds for condensing species with absolute shift.
    
    Physical basis: Condensational narrowing - adding the same absolute
    mass to all particles narrows the relative size distribution because
    the relative change is smaller for larger particles.
    """
    new_lower = np.copy(lower)
    new_upper = np.copy(upper)
    new_mean = np.copy(mean)
    
    # Shift bounds for condensing species by absolute amount
    new_lower[species_idx] = max(0, lower[species_idx] + delta_mass)
    new_upper[species_idx] = upper[species_idx] + delta_mass
    new_mean[species_idx] = mean[species_idx] + delta_mass
    
    # Other species bounds unchanged
    # Relative width naturally decreases: (upper-lower)/mean shrinks as mean grows
    return new_lower, new_upper, new_mean

# Example: Species with [90, 110] kg (±10%) gains 100 kg
# Result: [190, 210] kg (±5%) - relative width halved
```

**Coagulation (sampled with merge/keep)**:
1. Sample masses from both particle distributions (correlated across species)
2. Compute product mass distribution per species (min, mean, max of sums)
3. Calculate overlap with existing particles
4. If overlap > threshold (50%): merge via per-species mass-weighted bounds
5. Else: create new computational particle

**Bound combination (per-species, mass-weighted)**:
```python
def combine_bounds_coagulation(lower_a, upper_a, mean_a, lower_b, upper_b, mean_b):
    """Combine bounds from two coagulating particles, per species.
    
    Each species bounds combine independently using mass-weighted averaging
    of relative widths, preserving the variance information.
    """
    n_species = len(mean_a)
    product_mean = mean_a + mean_b
    product_lower = np.zeros(n_species)
    product_upper = np.zeros(n_species)
    
    for s in range(n_species):
        total_mass_s = mean_a[s] + mean_b[s]
        if total_mass_s > 0:
            weight_a = mean_a[s] / total_mass_s
            weight_b = mean_b[s] / total_mass_s
            rel_width_a = (upper_a[s] - lower_a[s]) / (2 * mean_a[s]) if mean_a[s] > 0 else 0
            rel_width_b = (upper_b[s] - lower_b[s]) / (2 * mean_b[s]) if mean_b[s] > 0 else 0
            product_rel_width = weight_a * rel_width_a + weight_b * rel_width_b
            # Clamp lower factor to avoid negative masses if product_rel_width > 1
            lower_factor = max(0.0, 1 - product_rel_width)
            product_lower[s] = total_mass_s * lower_factor
            product_upper[s] = total_mass_s * (1 + product_rel_width)
        else:
            product_lower[s] = 0
            product_upper[s] = 0
    
    return product_lower, product_upper, product_mean
```

**Split/merge maintenance**:
- Target: N computational particles
- Trigger when count < 0.75*N or count > N
- Split: Divide high-concentration particles
- Merge: Combine particles with highest distribution overlap

### Random State Management

```python
class ProbabilisticParticleResolved(DistributionStrategy):
    def __init__(self, seed=None, random_pool_size=10000):
        # Use a single RNG instance for all random draws to ensure reproducibility.
        # The random_pool_size argument is retained for backward compatibility but unused.
        self.rng = np.random.default_rng(seed)
    
    def _get_random_values(self, n: int) -> NDArray:
        # Draw directly from the RNG to avoid mid-operation pool refreshes that
        # can change the random sequence when call patterns change.
        return self.rng.random(n)
```

## Theoretical Considerations: Distribution Shape Choice

### Why Uniform Distribution (Current Design)

This implementation uses **uniform distributions** within mass bounds as a computational
approximation. This choice is deliberate and justified for v1, though future extensions
may consider alternative shapes.

### The Multiplicative Central Limit Theorem

The **multiplicative CLT** (sometimes called the "lognormal CLT") states that products
of independent positive random variables tend toward a **lognormal distribution**:

```
If X₁, X₂, ..., Xₙ are independent positive random variables with finite log-moments,
then:
    P_n = ∏ Xᵢ  →  Lognormal(nμ, nσ²)

where μ, σ² are the mean and variance of ln(Xᵢ).
```

This explains why **lognormal distributions emerge naturally in aerosol systems**:
- Coagulation is fundamentally **multiplicative** (mass roughly doubles per event)
- Fragmentation involves **multiplicative splitting**
- Growth processes compound **fractionally**

| Process | Mathematical Nature | Limiting Distribution |
|---------|---------------------|----------------------|
| Coagulation | Multiplicative | Lognormal |
| Condensation | Additive | Normal (via standard CLT) |
| Sum of uniforms | Additive | Irwin-Hall → Normal |
| Product of positives | Multiplicative | Lognormal |

### Implications for This Design

**Condensation (additive process):** The absolute bound shift approach (E4-F4) is
physically correct. Adding the same mass to all particles is additive, and the
standard CLT applies. The uniform approximation works well here, and the natural
"condensational narrowing" (relative width decreases as particles grow) is captured.

**Coagulation (multiplicative process):** Strictly speaking, the product of two
uniform distributions is **not** uniform—it follows a more complex distribution
that, over many events, tends toward lognormal. The current mass-weighted bound
averaging (E4-F5) is a **simplifying approximation** that:

1. Preserves computational efficiency (no shape parameter tracking)
2. Maintains correct **moments** (mean mass conserved)
3. Is accurate when distribution widths are narrow (±5-10%)
4. Avoids the complexity of tracking geometric standard deviation

### When the Uniform Approximation is Valid

The uniform approximation works well when:
- Relative distribution width is small (≤10%)
- Primary interest is in **moments** (mean, variance) rather than exact shape
- Number of coagulation events per computational particle is limited
- Computational simplicity is prioritized over distributional accuracy

### Future Extension Consideration

A future enhancement (beyond E4 scope) could add `LognormalMassDistribution` to
the `MassDistributionShape` interface, tracking geometric standard deviation and
using lognormal sampling for coagulation products. This would be appropriate for:
- Long-duration coagulation-dominated simulations
- Validation studies requiring accurate size distribution shapes
- Research into distribution evolution dynamics

For v1, the uniform approximation provides the best balance of accuracy, simplicity,
and computational efficiency for the target use cases (cloud microphysics, aerosol
coagulation with moderate event counts).

## Critical Testing Requirements

- **No Coverage Modifications**: Keep coverage thresholds at 80%+
- **Self-Contained Tests**: Each feature ships with `*_test.py` files
- **Test-First Completion**: Tests pass before phase completion
- **Mass Conservation**: Verify `E[total_mass]` is conserved in all dynamics
- **Statistical Validation**: Coagulation sampling produces correct distributions

## Testing Strategy

### Unit Tests (per feature)
- `particula/particles/distribution_strategies/tests/probabilistic_particle_resolved_test.py`
- `particula/particles/distribution_strategies/tests/mass_distribution_shape_test.py`
- `particula/particles/tests/representation_probabilistic_test.py`

### Integration Tests
- `particula/integration_tests/probabilistic_condensation_test.py`
- `particula/integration_tests/probabilistic_coagulation_test.py`

### Performance Benchmarks
- Compare computational cost vs `ParticleResolvedSpeciatedMass` at equivalent
  physical particle counts
- Validate O(n) scaling for condensation, O(n²) or O(n log n) for coagulation
- Target: 10x speedup at 100k physical particles with 1k computational particles

## Success Metrics

1. **Accuracy**: Coagulation kernel moments match analytical solutions within 5%
2. **Performance**: 10x speedup over particle-resolved at 100k particles
3. **Conservation**: Mass conserved to machine precision in condensation;
   E[mass] conserved in coagulation
4. **Usability**: Builder API matches existing representation patterns

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Coagulation sampling introduces bias | High | Validate against analytical kernels; increase sample count |
| Merge threshold too aggressive | Medium | Make threshold configurable; default conservative (50%) |
| Split/merge destabilizes simulations | Medium | Trigger only at thresholds; allow manual control |
| RNG state affects reproducibility | Low | Store seed; allow explicit RNG injection |

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-19 | Initial epic creation | ADW |
| 2026-01-19 | Changed to per-species bounds (2D arrays) with correlated sampling; condensation uses absolute shift to capture condensational narrowing | ADW |
| 2026-01-19 | Added theoretical considerations section documenting multiplicative CLT and uniform distribution approximation rationale | ADW |

## References

- Shima et al. (2009) "The super-droplet method for the numerical simulation
  of clouds and precipitation" - Super-droplet method foundation
- Unterstrasser et al. (2017) "Optimised microphysics" - Probabilistic sampling
  approaches
- Existing `ParticleResolvedSpeciatedMass` implementation in particula
- Multiplicative CLT / Lognormal emergence: Explains why aerosol size distributions
  are lognormal—multiplicative processes (coagulation, fragmentation) produce
  lognormal distributions via CLT applied in log-space. See Aitchison & Brown
  (1957) "The Lognormal Distribution" for theoretical foundations.
