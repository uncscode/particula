"""Compare emergent radiance from land and sea at glint and off-glint angles."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from disort import (
    DIRECT,
    HERE,
    VIEW_MU,
    WAVE,
    disort_slab,
    slab_depth,
    write_csv,
)
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import solve_shortwave
from particula.optics.solar import SolarInput
from particula.optics.surface import ocean_surface, ross_li_surface


def main(prefix):
    """Compare top radiance over the Ross–Li land and ocean surfaces."""
    tau = 0.1
    ssa = 0.92
    g = 0.55
    depth = slab_depth(tau)[None, :]
    column = SpectralColumn(
        WAVE,
        [1000.0, 0.0],
        depth,
        np.full_like(depth, ssa),
        np.tile(g ** np.arange(65), (1, 3, 1)),
    )
    sun = SolarInput(WAVE, VIEW_MU, DIRECT, 0.0 * DIRECT)
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        for name, surface, boundary in [
            (
                "land",
                ross_li_surface(WAVE, 0.22, 0.08, 0.015),
                (
                    "brdf_ambrals iso 0.22\nbrdf_ambrals vol 0.08\n"
                    "brdf_ambrals geo 0.015"
                ),
            ),
            ("ocean", ocean_surface(WAVE, wind_speed=5.0), "brdf_cam u10 5"),
        ]:
            for azimuth in [60.0, 170.0, 180.0]:
                ref = disort_slab(
                    prefix,
                    temp,
                    VIEW_MU,
                    0.0,
                    tau,
                    ssa,
                    g,
                    boundary=boundary,
                    azimuth=azimuth,
                )
                result = solve_shortwave(
                    column,
                    sun,
                    surface,
                    streams=16,
                    view_zenith_deg=np.degrees(np.arccos(VIEW_MU)),
                    relative_azimuth_deg=azimuth,
                )
                for j, wl in enumerate(WAVE):
                    rows.append(
                        dict(
                            surface=name,
                            relative_azimuth_deg=azimuth,
                            wavelength_nm=round(wl * 1e9),
                            disort_W_m2_sr_nm=ref[0, j, 3] * 1e-9,
                            native_W_m2_sr_nm=result.radiance[j] * 1e-9,
                            difference_pct=100
                            * (result.radiance[j] / ref[0, j, 3] - 1),
                        )
                    )
    write_csv(HERE / "directional.csv", rows)
    for name in ("land", "ocean"):
        print(
            name,
            min(r["difference_pct"] for r in rows if r["surface"] == name),
            max(r["difference_pct"] for r in rows if r["surface"] == name),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    main(parser.parse_args().prefix.resolve())
