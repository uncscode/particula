# particula.dynamics.condensation.mass_transfer_utils

*module*

Helper routines for condensation and evaporation mass-transfer
calculations.

These helpers isolate common logic from :pymod:`mass_transfer` so that each
computational step can be unit-tested independently.  All functions operate
on NumPy arrays describing either mass change or particle properties and
enforce physical limits such as available gas mass or particle inventory.

> **References**
>
> - P. Hinds, *Aerosol Technology*, 2nd ed., Wiley-Interscience, 1999.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer_utils.py#L1-L1)

## particula.dynamics.condensation.mass_transfer_utils.apply_condensation_limit

*function*

```python
def apply_condensation_limit(mass_to_change: NDArray[np.float64], gas_mass: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]
```

Limit condensation so that total uptake does not exceed available gas.

Positive mass changes (condensation) are summed per species and compared
to the available gas mass after accounting for simultaneous evaporation.
When the net uptake (condensation plus evaporation) would exceed the gas
inventory, only the positive portion is scaled to preserve conservation
while evaporation remains untouched.

**Parameters**

- `mass_to_change` (`NDArray[np.float64]`) — Requested mass change per bin and species in kg.
- `gas_mass` (`NDArray[np.float64]`) — Total gas mass available for condensation in kg.

**Returns**

- `mass_to_change_scaled` (`NDArray[np.float64]`) — Mass change array after applying the scaling factor for condensation.
- `evap_sum` (`NDArray[np.float64]`) — Column sum of evaporation (negative Δm) per species.
- `neg_mask` (`NDArray[np.bool_]`) — Boolean mask identifying evaporation elements.

**Examples**

```py title="Insufficient gas example"
scaled_dm, evap, mask = mtu.apply_condensation_limit(
    mass_to_change=np.array([[2.0e-9, -1.0e-9]]),
    gas_mass=np.array([5.0e-10])
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer_utils.py#L57-L112)

## particula.dynamics.condensation.mass_transfer_utils.apply_evaporation_limit

*function*

```python
def apply_evaporation_limit(mass_to_change: NDArray[np.float64], particle_mass: NDArray[np.float64], particle_concentration: NDArray[np.float64], evap_sum: NDArray[np.float64], neg_mask: NDArray[np.bool_]) -> NDArray[np.float64]
```

Limit evaporation so that mass loss honors the particle inventory.

The available inventory (I) per species is the sum of particle mass scaled
by particle concentration. When the requested evaporation exceeds the
inventory, only the negative entries are scaled so that no bin loses more
mass than exists, while condensation entries are left unchanged.

**Parameters**

- `mass_to_change` (`NDArray[np.float64]`) — Candidate mass change per bin/species in kg.
- `particle_mass` (`NDArray[np.float64]`) — Mass of one particle in each bin/species in kg.
- `particle_concentration` (`NDArray[np.float64]`) — Number concentration in m⁻³.
- `evap_sum` (`NDArray[np.float64]`) — Column sum of evaporation in kg (negative values).
- `neg_mask` (`NDArray[np.bool_]`) — Boolean mask identifying evaporation entries.

**Returns**

- (`NDArray[np.float64]`) — Mass change array with evaporation scaled to the available inventory.

**Examples**

```py title="Inventory-limited evaporation"
limited_dm = mtu.apply_evaporation_limit(
    mass_to_change, particle_mass, particle_conc,
    evap_sum, neg_mask
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer_utils.py#L115-L171)

## particula.dynamics.condensation.mass_transfer_utils.apply_per_bin_limit

*function*

```python
def apply_per_bin_limit(mass_to_change: NDArray[np.float64], particle_mass: NDArray[np.float64], particle_concentration: NDArray[np.float64]) -> NDArray[np.float64]
```

Ensure no bin loses more mass than it contains.

For each bin the maximum allowable evaporation is the particle mass per
bin multiplied by the bin concentration. Any requested mass change that
would deplete the bin beyond its inventory is clipped at that limit; this
guard follows the global gas and inventory scaling steps.

**Parameters**

- `mass_to_change` (`NDArray[np.float64]`) — Proposed mass change per bin/species in kg.
- `particle_mass` (`NDArray[np.float64]`) — Mass of a single particle per bin/species in kg.
- `particle_concentration` (`NDArray[np.float64]`) — Number concentration in m⁻³.

**Returns**

- (`NDArray[np.float64]`) — Mass change array after applying the per-bin evaporation limit.

**Examples**

```py title="Per-bin clipping"
clipped_dm = mtu.apply_per_bin_limit(
    mass_to_change, particle_mass, particle_conc
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer_utils.py#L174-L209)

## particula.dynamics.condensation.mass_transfer_utils.calc_mass_to_change

*function*

```python
def calc_mass_to_change(mass_rate: NDArray[np.float64], time_step: float | NDArray[np.float64], particle_concentration: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the requested mass change for every particle/species pair.

The instantaneous mass-transfer rate (ṁ) is integrated over a time
interval (Δt) and scaled by the particle number concentration (C) for
each bin and species:

Δm = ṁ × Δt × C

**Parameters**

- `mass_rate` (`NDArray[np.float64]`) — Mass transfer rate (ṁ) for each particle or ``(particle, species)`` pair in kg s⁻¹.
- `time_step` (`float | NDArray[np.float64]`) — Time step Δt in seconds (scalar or per-particle array).
- `particle_concentration` (`NDArray[np.float64]`) — Number concentration C in m⁻³.

**Returns**

- (`NDArray[np.float64]`) — Requested mass change Δm for every particle/species combination in kg.

**Examples**

```py title="Scalar rate"
import numpy as np
from particula.dynamics.condensation import mass_transfer_utils as mtu
mass_rate = np.array([1.0e-15])          # kg/s
dm = mtu.calc_mass_to_change(mass_rate, 10.0, np.array([1.0e6]))
# dm ≈ 1.0e-8 kg
```

> **References**
>
> - "Mass transfer",
>   [Wikipedia](https://en.wikipedia.org/wiki/Mass_transfer)

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer_utils.py#L17-L54)
