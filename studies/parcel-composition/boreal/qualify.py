"""Independent layered boundary-condition & sampled optical interpolation
checks.
"""

import argparse
from pathlib import Path

import numpy as np
from campaign import read
from scipy.special import spherical_jn, spherical_yn

from optics import Optics


def _boundary(bessel, order, argument, scale):
    value = bessel(order, argument)
    derivative = bessel(order, argument, True)
    return np.array([argument * value, scale * (value + argument * derivative)])


def multilayer(x, m):
    """Solve the two tangential field continuities per interface &
    polarization.
    """
    layers = len(m)
    exterior = []
    for n in range(1, round(x[-1] + 4 * x[-1] ** (1 / 3) + 8)):
        coeff = []
        for magnetic in (False, True):
            matrix = np.zeros((2 * layers, 2 * layers), complex)
            rhs = np.zeros(2 * layers, complex)
            for j, rad in enumerate(x):
                for k, sign in ((j, 1), (j + 1, -1)):
                    index = m[k] if k < layers else 1.0
                    z = index * rad

                    scale = index if magnetic else 1 / index
                    psi = _boundary(spherical_jn, n, z, scale)
                    rows = slice(2 * j, 2 * j + 2)
                    if k == layers:
                        # Exterior field = regular incident - outgoing
                        # scattered.
                        rhs[rows] = psi
                        matrix[rows, -1] = psi + 1j * _boundary(
                            spherical_yn, n, z, scale
                        )
                    elif k == 0:
                        matrix[rows, 0] = sign * psi
                    else:
                        matrix[rows, 2 * k - 1] = sign * psi
                        matrix[rows, 2 * k] = sign * _boundary(
                            spherical_yn, n, z, scale
                        )
            coeff.append(np.linalg.solve(matrix, rhs)[-1])
        exterior.append((2 * n + 1) * sum(v.real - abs(v) ** 2 for v in coeff))
    return 2 * sum(exterior) / x[-1] ** 2


def qualify(base, table):
    """Test a published five-layer geometry and sampled cloud optics."""
    from scattnlay import scattnlay

    m = np.array([1.8 + 1.7j, 0.8 + 0.7j, 1.2 + 0.09j, 2.8 + 0.2j, 1.5 + 0.4j])
    for x in [1.0, 2.0, 10.0]:
        rad = x * np.cbrt([0.1, 0.36, 0.404, 0.7706, 1.0])
        a = multilayer(rad, m)
        b = scattnlay(rad, m)[3]
        np.testing.assert_allclose(a, b, rtol=1e-9)
        print("five-layer", x, a, b)
    boxes, _, _ = read(base / "1199/c1b1n1/cloud-peak.npz")
    optic = Optics(table)
    for i, box in enumerate(boxes):
        idx = np.flatnonzero(box.dry[:, 3] > 0)[::29]
        v, w = box.dry[idx], box.water[idx]
        lookup = optic.areas(v, w)
        for j, water in enumerate((w * 0, w)):
            dry = v.sum(1)
            d = np.cbrt(dry + water)
            r = np.cbrt(v[:, 3] / (dry + water))
            shell = 1.5 - 0.167 * water / (dry - v[:, 3] + water)
            inputs = np.column_stack([d * r, d]) * np.pi / 550
            q = scattnlay(
                inputs, np.column_stack([np.full(len(v), 1.95 + 0.79j), shell])
            )[3]
            exact = np.pi * d**2 * q / 4
            print(
                "lookup",
                i,
                j,
                len(v),
                "integrated %",
                100 * (lookup[:, j].sum() / exact.sum() - 1),
                "particle max %",
                100 * max(abs(lookup[:, j] / exact - 1)),
            )
            if j == 0:
                np.testing.assert_allclose(
                    lookup[:, j].sum(), exact.sum(), rtol=0.005
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("table")
    a = p.parse_args()
    qualify(a.base, a.table)
