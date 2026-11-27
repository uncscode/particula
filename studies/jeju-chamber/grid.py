"""Finite-volume geometry of the prescribed circulation."""

import numpy as np
from flow import HEIGHT, WIDTH, Grid
from scipy.linalg import expm
from thermal import HeatModel


class ChamberGrid:
    """Coordinates are x, depth and height. Array order is height, depth, x."""

    def __init__(self, nx, ny, nz=3):
        self.shape = (nx, ny, nz)
        self.nx, self.nz = (
            nx * ny,
            nz,
        )  # horizontal plane size for legacy reductions
        self.size = nx * ny * nz
        self.dx, self.dy, self.dz = WIDTH / nx, WIDTH / ny, HEIGHT / nz
        iz, iy, ix = np.indices((nz, ny, nx)).reshape(3, -1)
        x, z = (
            (ix + 0.5) * self.dx,
            (iz + 0.5) * self.dz,
        )
        self.thermal = np.zeros(self.size)
        self.bottom = (iz == 0) / (nx * ny)
        self.port = self.bottom * np.exp(-0.5 * ((x - WIDTH / 2) / 0.45) ** 2)
        self.port /= self.port.sum()
        self.source = np.exp(
            -0.5
            * (
                ((x - 0.78 * WIDTH) / 0.35) ** 2
                + ((z - 0.72 * HEIGHT) / 0.48) ** 2
            )
        )
        self.source /= self.source.mean()
        self.faces = np.column_stack(
            [
                (ix == 0) * self.dy * self.dz,
                (ix == nx - 1) * self.dy * self.dz,
                (iy == 0) * self.dx * self.dz,
                (iy == ny - 1) * self.dx * self.dz,
                (iz == 0) * self.dx * self.dy,
                (iz == nz - 1) * self.dx * self.dy,
            ]
        )
        total = self.faces.sum(axis=0)
        np.testing.assert_allclose(total, [WIDTH * HEIGHT] * 4 + [WIDTH**2] * 2)
        self.upper = (iz == nz - 1) * np.exp(
            -0.5 * ((x - WIDTH / 2) / 0.60) ** 2
        )
        self.upper /= self.upper.sum()
        self.adv = np.zeros((self.size, self.size))
        self.cross = np.zeros_like(self.adv)
        self.diff = np.zeros_like(self.adv)
        labels = np.arange(self.size).reshape(nz, ny, nx)
        planar = Grid(nx, nz, "double")
        for layer in range(ny):
            k = labels[:, layer, :].ravel()
            self.adv[np.ix_(k, k)] += planar.advective
            self.diff[np.ix_(k, k)] += planar.diffusive
        if ny > 1:
            transverse = Grid(ny, nz, "single")
            for col in range(nx):
                k = labels[:, :, col].ravel()
                self.cross[np.ix_(k, k)] += transverse.advective
                # Add only depth exchange. Vertical exchange is already included.
                for j in range(nz):
                    for depth in range(ny - 1):
                        a, b = labels[j, depth : depth + 2, col]
                        self.diff[np.ix_([a, b], [a, b])] += (
                            np.array([[-1, 1], [1, -1]]) / self.dy**2
                        )

    def matrix(self, speed, diffusivity, dt, cross=0):
        generator = (
            speed * self.adv + diffusivity * self.diff + cross * self.cross
        )
        matrix = expm(generator * dt)
        if matrix.min() < -1e-13:
            raise ValueError("Negative transport coefficient")
        for axis in (0, 1):
            np.testing.assert_allclose(
                matrix.sum(axis=axis), 1, atol=1e-12, rtol=0
            )
        return matrix


class ChamberHeat(HeatModel):
    @staticmethod
    def geometry(grid, operator):
        # Volume-mean forcing is common to the two geometries.
        assert operator == "volume"
        return grid.faces, grid.upper, np.full(grid.size, 1 / grid.size)
