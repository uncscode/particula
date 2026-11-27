"""550 nm scattering by clear and BC-coated spheres in the redraw course."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
from particula.optics import get_mie_efficiencies
from scipy.ndimage import map_coordinates

from optics import BC, WAVE


def exact(d, core, index):
    """Return direct scattering efficiencies for bare or coated particles."""
    from scattnlay import scattnlay

    d, core, index = np.broadcast_arrays(d, core, index)
    x = d.ravel() * np.pi / WAVE
    r = core.ravel()
    m = index.astype(complex).ravel()
    out = np.empty(x.size)
    clear = r == 0
    if clear.any():
        out[clear] = scattnlay(x[clear, None], m[clear, None])[2]
    if (~clear).any():
        y = x[~clear]
        out[~clear] = scattnlay(
            np.column_stack((y * r[~clear], y)),
            np.column_stack((np.full(len(y), BC), m[~clear])),
        )[2]
    return out.reshape(d.shape)


def build(path):
    """Tabulate a log-diameter/core-ratio grid and check homogeneous limits."""
    if version("scattnlay") != "2.4":
        raise ValueError("Expected the study's pinned scattnlay 2.4")
    d = np.geomspace(1.4, 30000, 751)
    r = np.linspace(0, 1, 33)
    s = np.linspace(1.333, 1.50, 9)
    for idx, ratio, m in [(s[-1], 0, 1.5), (s[0], 0, 1.333), (s[-1], 1, BC)]:
        diam = np.geomspace(2, 22000, 25)
        q = get_mie_efficiencies(diam * 5e-10, WAVE * 1e-9, m).scattering
        np.testing.assert_allclose(
            exact(diam, ratio, idx), q, rtol=1e-7, atol=1e-13
        )
    q = np.stack(
        [np.column_stack([exact(d, ratio, idx) for ratio in r]) for idx in s]
    )
    if not np.isfinite(q).all() or q.min() <= 0:
        raise ValueError("Invalid scattering efficiencies")
    np.savez(path, d=d, r=r, s=s, q=np.log(q))


class Scattering:
    """Evaluate dry and ambient cross sections, including BC-free particles."""

    def __init__(self, path):
        """Load the uniform interpolation axes."""
        with np.load(path) as z:
            self.q = z["q"]
            self.logd = np.log(z["d"])
            self.shell = z["s"]

    def coordinates(self, dry, bc, water):
        """Derive diameter, core ratio and effective shell index."""
        v = dry + water
        d = np.cbrt(v)
        r = np.cbrt(bc / v)
        sw = np.divide(water, v - bc, out=np.zeros_like(v), where=v > bc)
        return d, r, 1.50 - 0.167 * sw

    def area(self, d, r, s):
        """Interpolate efficiency and convert projected area to nm²."""
        if (
            np.any(d < np.exp(self.logd[0]))
            or np.any(d > np.exp(self.logd[-1]))
            or np.any((r < 0) | (r > 1 + 1e-12))
        ):
            raise ValueError("Particle outside scattering grid")
        coords = np.array(
            [
                (s - self.shell[0])
                / (self.shell[-1] - self.shell[0])
                * (len(self.shell) - 1),
                (np.log(d) - self.logd[0])
                / (self.logd[-1] - self.logd[0])
                * (len(self.logd) - 1),
                r * (self.q.shape[2] - 1),
            ]
        )
        return (
            np.exp(map_coordinates(self.q, coords, order=1, mode="nearest"))
            * np.pi
            * d
            * d
            / 4
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    build(p.parse_args().path)
