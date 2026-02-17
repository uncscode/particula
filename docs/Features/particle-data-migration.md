---
title: Particle and Gas Data Migration Guide
---

# Particle and Gas Data Migration Guide

This guide describes how to migrate legacy APIs (`ParticleRepresentation`,
`GasSpecies`) to the new data containers (`ParticleData`, `GasData`). The
facades remain available for backward compatibility but emit
`DeprecationWarning` to encourage the new data-first workflow.

## Overview

The new containers separate **data** from **behavior**:

- `ParticleData` stores particle state in a batch-friendly layout.
- `GasData` stores gas species arrays with an explicit box dimension.
- `ParticleRepresentation` and `GasSpecies` remain as facades that wrap the
  new containers while preserving the legacy API surface.

The facades are deprecated. Existing code continues to work, but new code
should construct and pass the data containers directly.

## Quick Mapping

| Legacy API | New API | Notes |
| --- | --- | --- |
| `ParticleRepresentation` | `ParticleData` | Behavior stays in strategies/runnables |
| `GasSpecies` | `GasData` | Vapor-pressure strategies stay on the facade |

## GasSpecies → GasData

### Preferred (new) construction

```python
from particula.gas.gas_data import GasData
import numpy as np

gas_data = GasData(
    name=["Water"],
    molar_mass=np.array([0.018]),
    concentration=np.array([[1e-6]]),  # (n_boxes, n_species)
    partitioning=np.array([True]),
)
```

### Backward-compatible facade

```python
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import ConstantVaporPressureStrategy

species = GasSpecies(
    name="Water",
    molar_mass=0.018,
    vapor_pressure_strategy=ConstantVaporPressureStrategy(2330.0),
    concentration=1e-6,
)
```

The facade emits a `DeprecationWarning`. To access the data container from a
facade:

```python
gas_data = species.data
```

### Conversion helpers

```python
from particula.gas.gas_data import from_species, to_species

gas_data = from_species(species, n_boxes=1)
species = to_species(gas_data, [ConstantVaporPressureStrategy(2330.0)])
```

## ParticleRepresentation → ParticleData

The particle facade uses the same pattern. Construct `ParticleData` directly
for new code and use the `ParticleRepresentation` facade for legacy code.

## Handling Deprecation Warnings

If you need to silence deprecation warnings while migrating, filter them
explicitly in your application code:

```python
import warnings

warnings.filterwarnings(
    "ignore",
    message="ParticleRepresentation is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="GasSpecies is deprecated.*",
    category=DeprecationWarning,
)
```

## Additional Notes

- The facades assume a **single-box** view of data; `GasData` stores multiple
  boxes explicitly via its leading dimension.
- Vapor pressure strategies remain attached to the `GasSpecies` facade (they
  are behavior, not data). If you need to construct a facade from data without
  emitting a warning, use `GasSpecies.from_data(...)`.
