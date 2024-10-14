# Particle

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Particle

> Auto-generated documentation for [particula.particle](https://github.com/uncscode/particula/blob/main/particula/particle.py) module.

## Particle

[Show source in particle.py:386](https://github.com/uncscode/particula/blob/main/particula/particle.py#L386)

the Particle class!

#### Signature

```python
class Particle(ParticleWallLoss):
    def __init__(self, **kwargs): ...
```

#### See also

- [ParticleWallLoss](#particlewallloss)

### Particle()._coag_prep

[Show source in particle.py:410](https://github.com/uncscode/particula/blob/main/particula/particle.py#L410)

get all related quantities to coulomb enhancement

#### Signature

```python
def _coag_prep(self, other: "Particle"): ...
```

### Particle().coagulation

[Show source in particle.py:464](https://github.com/uncscode/particula/blob/main/particula/particle.py#L464)

Dimensioned particle--particle coagulation kernel

#### Signature

```python
def coagulation(self, other: "Particle" = None): ...
```

### Particle().coulomb_enhancement_continuum_limit

[Show source in particle.py:448](https://github.com/uncscode/particula/blob/main/particula/particle.py#L448)

 Continuum limit of Coulomb enhancement for particle--particle
coagulation.

#### Signature

```python
def coulomb_enhancement_continuum_limit(self, other: "Particle" = None): ...
```

### Particle().coulomb_enhancement_kinetic_limit

[Show source in particle.py:442](https://github.com/uncscode/particula/blob/main/particula/particle.py#L442)

 Kinetic limit of Coulomb enhancement for particle--particle
cooagulation.

#### Signature

```python
def coulomb_enhancement_kinetic_limit(self, other: "Particle" = None): ...
```

### Particle().coulomb_potential_ratio

[Show source in particle.py:437](https://github.com/uncscode/particula/blob/main/particula/particle.py#L437)

Calculates the Coulomb potential ratio.

#### Signature

```python
def coulomb_potential_ratio(self, other: "Particle" = None): ...
```

### Particle().diffusive_knudsen_number

[Show source in particle.py:454](https://github.com/uncscode/particula/blob/main/particula/particle.py#L454)

Diffusive Knudsen number.

#### Signature

```python
def diffusive_knudsen_number(self, other: "Particle" = None): ...
```

### Particle().dimensionless_coagulation

[Show source in particle.py:459](https://github.com/uncscode/particula/blob/main/particula/particle.py#L459)

Dimensionless particle--particle coagulation kernel.

#### Signature

```python
def dimensionless_coagulation(self, other: "Particle" = None): ...
```

### Particle().reduced_friction_factor

[Show source in particle.py:432](https://github.com/uncscode/particula/blob/main/particula/particle.py#L432)

Returns the reduced friction factor between two particles.

#### Signature

```python
def reduced_friction_factor(self, other: "Particle" = None): ...
```

### Particle().reduced_mass

[Show source in particle.py:427](https://github.com/uncscode/particula/blob/main/particula/particle.py#L427)

Returns the reduced mass.

#### Signature

```python
def reduced_mass(self, other: "Particle" = None): ...
```



## ParticleCondensation

[Show source in particle.py:255](https://github.com/uncscode/particula/blob/main/particula/particle.py#L255)

calculate some condensation stuff

#### Signature

```python
class ParticleCondensation(ParticleInstances):
    def __init__(self, **kwargs): ...
```

#### See also

- [ParticleInstances](#particleinstances)

### ParticleCondensation().condensation_redmass

[Show source in particle.py:280](https://github.com/uncscode/particula/blob/main/particula/particle.py#L280)

red mass

#### Signature

```python
def condensation_redmass(self): ...
```

### ParticleCondensation().fuchs_sutugin

[Show source in particle.py:298](https://github.com/uncscode/particula/blob/main/particula/particle.py#L298)

the fuchs-sutugin correction

#### Signature

```python
def fuchs_sutugin(self): ...
```

### ParticleCondensation().molecular_enhancement

[Show source in particle.py:272](https://github.com/uncscode/particula/blob/main/particula/particle.py#L272)

molecular enhancement

#### Signature

```python
def molecular_enhancement(self): ...
```

### ParticleCondensation().particle_growth

[Show source in particle.py:317](https://github.com/uncscode/particula/blob/main/particula/particle.py#L317)

particle growth in m/s

#### Signature

```python
def particle_growth(self): ...
```

### ParticleCondensation().particle_saturation_ratio

[Show source in particle.py:331](https://github.com/uncscode/particula/blob/main/particula/particle.py#L331)

Calculates the saturation ratio of the particle at its surface,
accounting for the Kelvin effect.

#### Returns

-------
float
    The saturation ratio of the particle at its surface.

#### Signature

```python
def particle_saturation_ratio(self): ...
```

### ParticleCondensation().vapor_flux

[Show source in particle.py:305](https://github.com/uncscode/particula/blob/main/particula/particle.py#L305)

vapor flux

#### Signature

```python
def vapor_flux(self): ...
```

### ParticleCondensation().vapor_speed

[Show source in particle.py:289](https://github.com/uncscode/particula/blob/main/particula/particle.py#L289)

vapor speed

#### Signature

```python
def vapor_speed(self): ...
```



## ParticleDistribution

[Show source in particle.py:34](https://github.com/uncscode/particula/blob/main/particula/particle.py#L34)

starting a particle distribution from continuous pdf

#### Signature

```python
class ParticleDistribution(Vapor):
    def __init__(self, **kwargs): ...
```

#### See also

- [Vapor](./vapor.md#vapor)

### ParticleDistribution().pre_discretize

[Show source in particle.py:106](https://github.com/uncscode/particula/blob/main/particula/particle.py#L106)

Returns a distribution pdf of the particles

Utilizing the utility discretize to get make a lognorm distribution
via scipy.stats.lognorm.pdf:
    interval: the size interval of the distribution
    gsigma  : geometric standard deviation of distribution
    mode    : geometric mean radius of the particles

#### Signature

```python
def pre_discretize(self): ...
```

### ParticleDistribution().pre_distribution

[Show source in particle.py:124](https://github.com/uncscode/particula/blob/main/particula/particle.py#L124)

Returns a distribution pdf of the particles

Utilizing the utility discretize to get make a lognorm distribution
via scipy.stats.lognorm.pdf:
    interval: the size interval of the distribution
    gsigma  : geometric standard deviation of distribution
    mode    : geometric mean radius of the particles

#### Signature

```python
def pre_distribution(self): ...
```

### ParticleDistribution().pre_radius

[Show source in particle.py:60](https://github.com/uncscode/particula/blob/main/particula/particle.py#L60)

Returns the radius space of the particles

Utilizing the utility cut_rad to get 99.99% of the distribution.
From this interval, radius is made on a linspace with nbins points.
Note: linspace is used here to practical purposes --- often, the
logspace treatment will return errors in the discretization due
to the asymmetry across the interval (finer resolution for smaller
particles, but much coarser resolution for larger particles).

#### Signature

```python
def pre_radius(self): ...
```



## ParticleInstances

[Show source in particle.py:137](https://github.com/uncscode/particula/blob/main/particula/particle.py#L137)

starting a particle distribution from single particles

#### Signature

```python
class ParticleInstances(ParticleDistribution):
    def __init__(self, **kwargs): ...
```

#### See also

- [ParticleDistribution](#particledistribution)

### ParticleInstances().aerodynamic_mobility

[Show source in particle.py:237](https://github.com/uncscode/particula/blob/main/particula/particle.py#L237)

Returns a particle's aerodynamic mobility.

#### Signature

```python
def aerodynamic_mobility(self): ...
```

### ParticleInstances().diffusion_coefficient

[Show source in particle.py:246](https://github.com/uncscode/particula/blob/main/particula/particle.py#L246)

Returns a particle's diffusion coefficient.

#### Signature

```python
def diffusion_coefficient(self): ...
```

### ParticleInstances().friction_factor

[Show source in particle.py:218](https://github.com/uncscode/particula/blob/main/particula/particle.py#L218)

Returns a particle's friction factor.

#### Signature

```python
def friction_factor(self): ...
```

### ParticleInstances().knudsen_number

[Show source in particle.py:202](https://github.com/uncscode/particula/blob/main/particula/particle.py#L202)

Returns particle's Knudsen number.

#### Signature

```python
def knudsen_number(self): ...
```

### ParticleInstances().particle_area

[Show source in particle.py:194](https://github.com/uncscode/particula/blob/main/particula/particle.py#L194)

Returns particle's surface area

#### Signature

```python
def particle_area(self): ...
```

### ParticleInstances().particle_distribution

[Show source in particle.py:175](https://github.com/uncscode/particula/blob/main/particula/particle.py#L175)

distribution

#### Signature

```python
def particle_distribution(self): ...
```

### ParticleInstances().particle_mass

[Show source in particle.py:184](https://github.com/uncscode/particula/blob/main/particula/particle.py#L184)

Returns mass of particle.

#### Signature

```python
def particle_mass(self): ...
```

### ParticleInstances().settling_velocity

[Show source in particle.py:227](https://github.com/uncscode/particula/blob/main/particula/particle.py#L227)

Returns a particle's settling velocity.

#### Signature

```python
def settling_velocity(self): ...
```

### ParticleInstances().slip_correction_factor

[Show source in particle.py:210](https://github.com/uncscode/particula/blob/main/particula/particle.py#L210)

Returns particle's Cunningham slip correction factor.

#### Signature

```python
def slip_correction_factor(self): ...
```



## ParticleWallLoss

[Show source in particle.py:354](https://github.com/uncscode/particula/blob/main/particula/particle.py#L354)

continuing...

#### Signature

```python
class ParticleWallLoss(ParticleCondensation):
    def __init__(self, **kwargs): ...
```

#### See also

- [ParticleCondensation](#particlecondensation)

### ParticleWallLoss().wall_loss_coefficient

[Show source in particle.py:374](https://github.com/uncscode/particula/blob/main/particula/particle.py#L374)

Returns a particle's wall loss coefficient.

#### Signature

```python
def wall_loss_coefficient(self): ...
```
