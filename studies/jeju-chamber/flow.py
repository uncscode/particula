"""Balanced equal-volume circulation and conservative cohort transport."""

import numpy as np
import warp as wp
from scipy.linalg import expm

HEIGHT = 3.0
WIDTH = (22.4 / HEIGHT) ** 0.5


class Grid:
    """A vertical cross section of the volume-equivalent rectangular chamber."""

    def __init__(self, nx, nz, topology="double"):
        if topology not in {"double", "single"}:
            raise ValueError(f"Unknown circulation topology {topology!r}")
        self.topology = topology
        self.nx, self.nz = nx, nz
        self.size = nx * nz
        self.dx, self.dz = WIDTH / nx, HEIGHT / nz
        self.x = (np.arange(nx) + 0.5) * self.dx
        self.z = (np.arange(nz) + 0.5) * self.dz
        x, z = (a.ravel() for a in np.meshgrid(self.x, self.z))
        cool = 0.65 * (abs(x - WIDTH / 2) / (WIDTH / 2)) ** 1.7
        cool += 0.35 * (1 - z / HEIGHT)
        cool = cool.mean() - cool
        self.thermal = cool / max(-cool.min(), 1.0e-12)
        self.bottom = (z < self.dz).astype(float) / nx
        self.port = self.bottom * np.exp(-0.5 * ((x - WIDTH / 2) / 0.45) ** 2)
        self.port /= self.port.sum()
        # Fixed, grid-independent footprint for the hypothesized upper-side source.
        salt = ((x - 0.78 * WIDTH) / 0.35) ** 2 + (
            (z - 0.72 * HEIGHT) / 0.48
        ) ** 2
        self.source = np.exp(-0.5 * salt)
        self.source /= self.source.mean()
        self.advective, self.diffusive = self._generators()

    def _generators(self):
        """Construct a discrete streamfunction and symmetric face exchange."""
        nx, nz = self.nx, self.nz
        advection = np.zeros((self.size, self.size))
        diffusion = np.zeros_like(advection)
        if nx == 1 and nz == 1:
            return advection, diffusion
        x = np.linspace(0, 1, nx + 1)
        z = np.linspace(0, 1, nz + 1)
        harmonic = 2 if self.topology == "double" else 1
        psi = np.outer(np.sin(np.pi * z), -np.sin(harmonic * np.pi * x))
        psi[[0, -1]] = 0
        psi[:, [0, -1]] = 0
        # Speed knob is metres per second. Streamfunction units are m²/s.
        psi *= HEIGHT / np.pi

        def face(i, j, flux, exchange):
            origin, end = (i, j) if flux >= 0 else (j, i)
            advection[end, origin] += abs(flux) / (self.dx * self.dz)
            diffusion[j, i] += exchange
            diffusion[i, j] += exchange

        for row in range(nz):
            for col in range(nx):
                i = row * nx + col
                if col + 1 < nx:
                    face(
                        i,
                        i + 1,
                        psi[row + 1, col + 1] - psi[row, col + 1],
                        self.dx**-2,
                    )
                if row + 1 < nz:
                    face(
                        i,
                        i + nx,
                        psi[row + 1, col] - psi[row + 1, col + 1],
                        self.dz**-2,
                    )
        for m in (advection, diffusion):
            m[np.diag_indices(self.size)] -= m.sum(axis=0)
            np.testing.assert_allclose(m.sum(axis=1), 0, atol=1e-12)
        return advection, diffusion

    def matrix(self, speed, diffusivity, dt):
        """Exact transport over one interval with fixed face fluxes."""
        a = self.advective if speed >= 0 else self.advective.T
        return expm(dt * (abs(speed) * a + diffusivity * self.diffusive))

    def match_temperature_and_sampling(self, coarse):
        """Remap the coarse thermal field and physical inlet depth by overlap."""

        def overlap(total, nfine, ncoarse):
            f = np.linspace(0, total, nfine + 1)
            c = np.linspace(0, total, ncoarse + 1)
            left = np.maximum(f[:-1, None], c[None, :-1])
            right = np.minimum(f[1:, None], c[None, 1:])
            return np.maximum(0, right - left) / (total / nfine)

        horizontal = overlap(WIDTH, self.nx, coarse.nx)
        vertical = overlap(HEIGHT, self.nz, coarse.nz)
        shape = coarse.thermal.reshape(coarse.nz, coarse.nx)
        self.thermal = (vertical @ shape @ horizontal.T).ravel()
        bottom = np.broadcast_to(vertical[:, :1], (self.nz, self.nx))
        self.bottom = (bottom / bottom.sum()).ravel().copy()
        footprint = np.exp(-0.5 * ((self.x - WIDTH / 2) / 0.45) ** 2)
        port = bottom * footprint
        self.port = (port / port.sum()).ravel()
        return self


@wp.kernel
def mix_particles(
    original: wp.array3d(dtype=float),
    mixed: wp.array3d(dtype=float),
    weights: wp.array3d(dtype=float),
    nl: int,
    ni: int,
    cells: int,
):
    """Exchange like-label particle moments while conserving their inventories."""
    job, slot = wp.tid()
    cell = job % cells
    case = job // (cells * 6)
    if weights.shape[0] == original.shape[0] // cells:
        case = job // cells
    group = job - cell
    # Two independent inventories use the same transport. Ice has fixed volumes.
    if slot < nl:
        n = float(0)
        v = float(0)
        dry = float(0)
        activation = float(0)
        for j in range(cells):
            origin = group + j
            take = weights[case, cell, j] * original[origin, 0, slot]
            n += take
            v += take * original[origin, 1, slot]
            dry += take * original[origin, 2, slot]
            activation += take * original[origin, 3, slot]
        mixed[job, 0, slot] = n
        if n > 0.0:
            mixed[job, 1, slot] = v / n
            mixed[job, 2, slot] = dry / n
            mixed[job, 3, slot] = activation / n
        else:
            mixed[job, 1, slot] = original[job, 1, slot]
            mixed[job, 2, slot] = original[job, 2, slot]
            mixed[job, 3, slot] = original[job, 3, slot]
    if slot < ni:
        ice = float(0)
        for j in range(cells):
            ice += weights[case, cell, j] * original[group + j, 4, slot]
        mixed[job, 4, slot] = ice


@wp.kernel
def mix_air(
    original: wp.array2d(dtype=float),
    mixed: wp.array2d(dtype=float),
    weights: wp.array3d(dtype=float),
    cells: int,
):
    """Advect vapor and the effective activation/contact histories."""
    job, field = wp.tid()
    cell = job % cells
    case = job // (cells * 6)
    if weights.shape[0] == original.shape[0] // cells:
        case = job // cells
    if field == 7:
        mixed[job, field] = original[job, field]
        return
    total = float(0)
    for j in range(cells):
        total += weights[case, cell, j] * original[job - cell + j, field]
    mixed[job, field] = total
