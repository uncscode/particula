"""Load a local MODIS Collection 6.1 land BRDF pixel with band quality.

MCD43A1 supplies iso/vol/geo coefficients. The matched MCD43A2 tile
supplies the full-inversion quality flag for each band. Provider specs:
https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD43A1
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
    """Read one local A1/A2 tile pixel as a Ross–Li lower boundary.

    Row and column are zero-based tile indices. Assign a MODIS band (1–7)
    to each modeled wavelength explicitly. The user owns interpolation or
    extrapolation between observed bands. Only full inversions (QA 0 or 1)
    with valid packed coefficients are accepted. Requires optional pyhdf.
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
