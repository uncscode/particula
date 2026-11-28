"""Read matching MODIS HDF4 parameter and quality tiles without network I/O."""

import numpy as np
import pytest
from particula.data.mcd43 import _tile, read_mcd43_surface


def test_local_mcd43_tile(tmp_path):
    """Exercise HDF field layout, scale, QA, dates, and spectral mapping."""
    sd = pytest.importorskip("pyhdf.SD")
    paths = [
        tmp_path / f"{product}.A2001317.h21v06.061.1.hdf"
        for product in ("MCD43A1", "MCD43A2")
    ]
    with _tile(paths[0], sd.SDC.WRITE | sd.SDC.CREATE) as tile:
        for i in (1, 2):
            field = tile.create(
                f"BRDF_Albedo_Parameters_Band{i}", sd.SDC.INT16, (2, 2, 3)
            )
            field[:] = np.full((2, 2, 3), i * 50, dtype=np.int16)
            field.endaccess()
    with _tile(paths[1], sd.SDC.WRITE | sd.SDC.CREATE) as tile:
        for i in (1, 2):
            field = tile.create(
                f"BRDF_Albedo_Band_Quality_Band{i}", sd.SDC.UINT8, (2, 2)
            )
            field[:] = np.array([[1, 0], [2, 255]], dtype=np.uint8)
            field.endaccess()
    options = dict(wavelength_m=[645e-9, 858e-9], band_ids=[1, 2])
    surface = read_mcd43_surface(*paths, row=0, col=0, **options)
    np.testing.assert_allclose(
        surface.coefficients, np.tile([0.05, 0.1], (3, 1))
    )
    with pytest.raises(ValueError, match="full inversion unavailable"):
        read_mcd43_surface(*paths, row=1, col=0, **options)
    with pytest.raises(ValueError, match="matching tile"):
        read_mcd43_surface(
            paths[0],
            tmp_path / "MCD43A2.A2001333.h21v06.061.1.hdf",
            row=0,
            col=0,
            **options,
        )
