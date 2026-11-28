# particula.data.mcd43

*module*

Load a local MODIS Collection 6.1 land BRDF pixel with band quality.

MCD43A1 supplies isotropic, volumetric and geometric coefficients. MCD43A2
supplies the retrieval quality flag for each band. See the Collection 6.1
product specifications at
https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A1 and
https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A2.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/mcd43.py#L1-L1)

## particula.data.mcd43._identity

*function*

```python
def _identity(path, product)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/mcd43.py#L31-L38)

## particula.data.mcd43._pixel

*function*

```python
def _pixel(dataset, field, row, col, with_parameters=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/mcd43.py#L41-L49)

## particula.data.mcd43._tile

*function*

```python
def _tile(path, mode)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/mcd43.py#L20-L28)

## particula.data.mcd43.read_mcd43_surface

*function*

```python
def read_mcd43_surface(parameters_path, quality_path, *, row, col, wavelength_m, band_ids)
```

Read a MODIS Collection 6.1 land BRDF for one pixel in matched tiles.

**Parameters**

- `parameters_path` (`str | Path`) — Local native MCD43A1 HDF4 granule with packed RossThick/LiSparseReciprocal parameters.
- `quality_path` (`str | Path`) — Local MCD43A2 HDF4 granule with the same tile and retrieval date as the parameters granule.
- `row` (`int`) — Zero-based integer pixel row within the tile.
- `col` (`int`) — Zero-based integer pixel column within the tile.
- `wavelength_m` (`ArrayLike`) — Increasing positive modeled vacuum wavelengths [m].
- `band_ids` (`ArrayLike`) — One MODIS band number (integer 1–7) for each wavelength. This mapping assigns retrieved coefficients directly to the grid.

**Returns**

- (`Surface`) — The isotropic, volumetric and geometric coefficients after applying the 0.001 packing scale. Its source records the files, pixel and per-band full-inversion quality flags.

**Raises**

- `ImportError` — The optional HDF4 reader pyhdf is unavailable.
- `ValueError` — The filenames or pixel/band mapping are invalid, a band
has a quality flag other than 0 or 1, or a packed coefficient
is outside its valid range.
- `HDF4Error` — The HDF4 reader cannot access a file or required dataset.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/mcd43.py#L52-L131)
