"""Pinned concentric-sphere absorption tables in air at 550 nm."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
from particula.optics import get_mie_efficiencies
from scipy.interpolate import RegularGridInterpolator

BC = 1.95 + 0.79j
WAVE = 550.0


def direct(d, ratio, shell):
    """Return exact two-layer absorption efficiency, inputs in nm."""
    from scattnlay import scattnlay

    x = np.pi * np.asarray(d) / WAVE
    xr, r = np.broadcast_arrays(x, ratio)
    radii = np.column_stack([(xr * r).ravel(), xr.ravel()])
    return np.asarray(
        scattnlay(radii, np.array([BC, complex(shell)]))[3]
    ).reshape(xr.shape)


def build(path):
    """Qualify direct Mie and tabulate absorption per absorbing volume."""
    from scattnlay import scattnlay

    if version("scattnlay") != "2.4":
        raise ValueError("Use the qualified scattnlay 2.4 distribution")
    diameter = np.geomspace(1.45, 30000.0, 281)
    ratio = np.geomspace(0.0005, 1.0, 83)
    shell = np.linspace(1.333, 1.50, 5)
    # A fictitious internal boundary in a homogeneous sphere has no effect.
    for x in [0.055, 1.0, 10.0, 100.0]:
        for m in [1.5 + 1j, BC]:
            layered = scattnlay(np.array([0.63 * x, x]), np.array([m, m]))[3]
            bare = get_mie_efficiencies(
                x * WAVE * 1e-9 / (2 * np.pi), WAVE * 1e-9, m
            ).absorption
            np.testing.assert_allclose(layered, bare, atol=1e-11, rtol=1e-9)
    table = []
    for index in shell:
        rows = []
        for d in diameter:
            q = direct(d, ratio, index)
            rows.append(q / ratio**3)
        table.append(rows)
    table = np.asarray(table)
    if not np.isfinite(table).all() or table.min() < -1e-4:
        raise ValueError(f"unstable multilayer table: {table.min()}")
    # Roundoff in extinction minus scattering is visible at vanishing cores.
    table = np.maximum(table, 0)
    hom = np.stack(
        [
            get_mie_efficiencies(
                d * 5e-10, WAVE * 1e-9, 1.50 + (BC - 1.50) * ratio**3
            ).absorption
            / ratio**3
            for d in diameter[diameter < 3000]
        ]
    )
    np.savez(path, d=diameter, r=ratio, s=shell, table=table, hom=hom)


class Optics:
    """Use bounded interpolation; clear particles have zero absorption."""

    def __init__(self, path):
        """Load the independently qualified coated and homogeneous tables."""
        t = np.load(path)
        self.coated = RegularGridInterpolator(
            (t["s"], np.log(t["d"]), np.log(t["r"])), t["table"]
        )
        self.hom = RegularGridInterpolator(
            (np.log(t["d"][t["d"] < 3000]), np.log(t["r"])), t["hom"]
        )

    def areas(self, volume, water):
        """Dry coated, wet coated and dry homogeneous absorption area in nm²."""
        total = volume.sum(1)
        bc = volume[:, 3]
        active = bc > 0
        out = np.zeros((len(total), 3))
        dry = total[active]
        wet = water[active]
        for i, liquid in enumerate([np.zeros_like(wet), wet]):
            d = np.cbrt(dry + liquid)
            r = np.cbrt(bc[active] / (dry + liquid))
            index = 1.50 - 0.167 * np.divide(
                liquid,
                dry - bc[active] + liquid,
                out=np.zeros_like(liquid),
                where=(dry - bc[active] + liquid) > 0,
            )
            area = np.pi * d**2 * r**3 / 4
            out[active, i] = area * self.coated(
                (index, np.log(d), np.log(np.maximum(r, 0.0005)))
            )
            if i == 0:
                out[active, 2] = area * self.hom(
                    (np.log(d), np.log(np.maximum(r, 0.0005)))
                )
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    build(parser.parse_args().path)
