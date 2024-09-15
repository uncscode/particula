# Relational Maps

## Particles

``` mermaid
mindmap
  root((*particula.next.particles*))
    distribution
      strategies
        MassBasedMovingBin
        RadiiBasedMovingBin
        SpeciatedMassMovingBin
        ParticleResolvedSpeciatedMass
      builders
      DistributionFactory
    activity
      strategies
        ActivityIdealMass
        ActivityIdealMolar
        ActivityKappaParameter
      builders
      ActivityFactory
    representation
      ParticleRepresentation
      builders
      ParticleRepresentationFactory
    surface
      strategies
        SurfaceStrategyVolume
        SurfaceStrategyMass
        SurfaceStrategyMolar
      builders
      SurfaceFactory
    properties
```