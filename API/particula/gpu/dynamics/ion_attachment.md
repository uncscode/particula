# particula.gpu.dynamics.ion_attachment

*module*

LF2013 attachment on the device using conditioned published coefficients.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_attachment.py#L1-L1)

## particula.gpu.dynamics.ion_attachment._CUTOFF

*attribute* · *module attribute*

```python
_CUTOFF = wp.constant(wp.float64(1e-15))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_attachment.py#L9-L9)

## particula.gpu.dynamics.ion_attachment._MAX_RADIUS

*attribute* · *module attribute*

```python
_MAX_RADIUS = wp.constant(wp.float64(1e-05))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_attachment.py#L8-L8)

## particula.gpu.dynamics.ion_attachment._MIN_RADIUS

*attribute* · *module attribute*

```python
_MIN_RADIUS = wp.constant(wp.float64(2e-10))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_attachment.py#L7-L7)

## particula.gpu.dynamics.ion_attachment.classify_ion_particle_wp

*function*

```python
def classify_ion_particle_wp(masses: wp.array3d(dtype=wp.float64), charges: wp.array2d(dtype=wp.float64), box: wp.int32, slot: wp.int32, positive_species: wp.int32, negative_species: wp.int32, positive_mass: wp.float64, negative_mass: wp.float64) -> wp.int32
```

Classify active particles with reserved ion material lanes.

Return 0 for aerosol, +1/-1 for a free ion, 2 for an inert neutral
recombination product, or 3 for unsupported ion-only composition/charge.
The caller must validate species indices, positive monomer masses and
native particle arrays. This function is only called for active slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_attachment.py#L66-L101)

## particula.gpu.dynamics.ion_attachment.ion_attachment_wp

*function*

```python
def ion_attachment_wp(radius: wp.float64, charge: wp.float64, polarity: wp.int32, omitted: wp.int32, tables: wp.array3d(dtype=wp.float64)) -> wp.float64
```

Return m^3/s, or -1 for invalid/unsupported input requiring a fault.

Omitted policy: 0 rejects; 1 returns the zero lower bound; 2 returns the
published cutoff upper bound. Bounds apply only to omitted repulsive
rates. The caller must validate the fixed temperature/material contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_attachment.py#L12-L63)

## particula.gpu.dynamics.ion_attachment.ion_collision_rate_wp

*function*

```python
def ion_collision_rate_wp(first_kind: wp.int32, second_kind: wp.int32, first_radius: wp.float64, second_radius: wp.float64, first_charge: wp.float64, second_charge: wp.float64, recombination: wp.float64, omitted: wp.int32, tables: wp.array3d(dtype=wp.float64)) -> wp.float64
```

Rate for pairs with at least one ion/product; -1 requires a fault.

Neutral ion-only products are inert; same-sign ions do not cluster.
Opposite free ions use the configured recombination coefficient [m^3/s].
Ion-aerosol pairs use LF2013. Aerosol-aerosol pairs must be handled by the
caller's charged Brownian model and reject here to prevent double counting.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/ion_attachment.py#L104-L142)
