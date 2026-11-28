"""Load a local MODIS Collection 6.1 land BRDF pixel with band quality.

MCD43A1 supplies isotropic, volumetric and geometric coefficients. MCD43A2
supplies the retrieval quality flag for each band. See the Collection 6.1
product specifications at
https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A1 and
https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A2.
"""

import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from particula.optics._spectral import wavelengths
from particula.optics.surface import ross_li_surface


@contextmanager
def _tile(path, mode):
    from pyhdf.SD import SD

    tile = SD(str(path), mode)
    try:
        yield tile
    finally:
        tile.end()


def _identity(path, product):
    match = re.fullmatch(
        rf"{product}\.A(\d{{7}})\.(h\d{{2}}v\d{{2}})\.061\.\d+\.hdf",
        path.name,
    )
    if match is None:
        raise ValueError(f"Expected a native Collection 6.1 {product} tile")
    return match.groups()


def _pixel(dataset, field, row, col, with_parameters=False):
    sds = dataset.select(field)
    try:
        dims = sds.info()[2]
        if row >= dims[0] or col >= dims[1]:
            raise ValueError("Pixel lies outside the MODIS tile")
        return sds[row, col, :] if with_parameters else sds[row, col]
    finally:
        sds.endaccess()


def read_mcd43_surface(
    parameters_path, quality_path, *, row, col, wavelength_m, band_ids
):
    """Read a MODIS Collection 6.1 land BRDF for one pixel in matched tiles.

    Args:
        parameters_path (str | Path): Local native MCD43A1 HDF4 granule with
            packed RossThick/LiSparseReciprocal parameters.
        quality_path (str | Path): Local MCD43A2 HDF4 granule with the same
            tile and retrieval date as the parameters granule.
        row (int): Zero-based integer pixel row within the tile.
        col (int): Zero-based integer pixel column within the tile.
        wavelength_m (ArrayLike): Increasing positive modeled vacuum
            wavelengths [m].
        band_ids (ArrayLike): One MODIS band number (integer 1–7) for each
            wavelength. This mapping assigns retrieved coefficients directly to
            the grid.

    Returns:
        (Surface): The isotropic, volumetric and geometric coefficients after
            applying the 0.001 packing scale. Its source records the files,
            pixel and per-band full-inversion quality flags.

    Raises:
        ImportError: The optional HDF4 reader pyhdf is unavailable.
        ValueError: The filenames or pixel/band mapping are invalid, a band
            has a quality flag other than 0 or 1, or a packed coefficient
            is outside its valid range.
        HDF4Error: The HDF4 reader cannot access a file or required dataset.
    """
    from pyhdf.SD import SDC

    pfile, qfile = Path(parameters_path), Path(quality_path)
    if _identity(pfile, "MCD43A1") != _identity(qfile, "MCD43A2"):
        raise ValueError("Require a matching tile and retrieval date")
    wave = wavelengths(wavelength_m)
    bands = np.asarray(band_ids)
    if (
        bands.dtype.kind not in "iu"
        or bands.shape != wave.shape
        or np.any((bands < 1) | (bands > 7))
    ):
        raise ValueError("Specify a MODIS band 1–7 per wavelength")
    if any(not isinstance(v, (int, np.integer)) or v < 0 for v in (row, col)):
        raise ValueError("Pixel indices must be nonnegative integers")
    coefficients = np.empty((len(bands), 3))
    quality_flags = {}
    with (
        _tile(pfile, SDC.READ) as params,
        _tile(qfile, SDC.READ) as quality,
    ):
        for band in np.unique(bands):
            flag = int(
                _pixel(
                    quality, f"BRDF_Albedo_Band_Quality_Band{band}", row, col
                )
            )
            if flag not in (0, 1):
                raise ValueError(
                    f"Band {band} full inversion unavailable (QA {flag})"
                )
            packed = np.asarray(
                _pixel(
                    params, f"BRDF_Albedo_Parameters_Band{band}", row, col, True
                )
            )
            if packed.shape != (3,) or np.any((packed < 0) | (packed > 32766)):
                raise ValueError(
                    f"Missing or invalid BRDF parameters in band {band}"
                )
            coefficients[bands == band] = packed * 0.001
            quality_flags[int(band)] = flag
    return ross_li_surface(
        wave,
        *coefficients.T,
        source=(
            f"{pfile.name} and {qfile.name}, 16-day retrieval, "
            f"pixel ({row}, {col}), band QA {quality_flags}"
        ),
    )
