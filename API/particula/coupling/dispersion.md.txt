# particula.coupling.dispersion

*module*

Flat, reflecting-ground reference solutions for passive aerosol mass.

The steady plume uses Briggs open-country spreads (PNNL-20990, Table A-1).
The Gaussian puff solves constant-diffusivity advection and diffusion. A
uniform ellipsoid provides a finite-width alternative with the same free-air
second moments. All three assume uniform horizontal wind and a ground image.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/dispersion.py#L1-L1)

## particula.coupling.dispersion._BRIGGS

*attribute* · *module attribute*

```python
_BRIGGS = {'A': (0.22, 0.2, 0.0, 0.0), 'B': (0.16, 0.12, 0.0, 0.0), 'C': (0.11, 0.08, 0.0002, -0.5), 'D': (0.08, 0.06, 0.0015, -0.5), 'E': (0.06, 0.03, 0.0003, -1.0), 'F': (0.04, 0.016, 0.0003, -1.0)}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/dispersion.py#L12-L19)

## particula.coupling.dispersion._coordinates

*function*

```python
def _coordinates(x, y, z, height)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/dispersion.py#L22-L30)

## particula.coupling.dispersion._normal

*function*

```python
def _normal(distance, sigma)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/dispersion.py#L33-L34)

## particula.coupling.dispersion.gaussian_plume

*function*

```python
def gaussian_plume(x, y, z, *, rate, wind, height, stability)
```

Return kg/m³ at downwind x, crosswind y and height z (metres).

rate is the continuous source (kg/s), wind the downwind speed (m/s),
height the effective source height (m), and stability the Pasquill class
A–F. The point-source plane and upwind half-space return zero.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/dispersion.py#L37-L64)

## particula.coupling.dispersion.puff_concentration

*function*

```python
def puff_concentration(x, y, z, *, mass, age, wind, height, diffusivity, shape='gaussian')
```

Return kg/m³ from one instantaneous release at (0, 0, height) in m.

mass is kg, age is seconds after release, wind is signed x velocity
(m/s), and diffusivity gives positive x, y, z diffusivities (m²/s).
A top_hat has uniform density inside an ellipsoid of semi-axes
sqrt(5) times the Gaussian widths, so both have variance 2 K t in
free air. Ground reflection folds the vertical distribution.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/dispersion.py#L67-L101)
