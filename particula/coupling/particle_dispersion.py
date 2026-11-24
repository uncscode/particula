"""Resolved particle transport using prescribed eddy diffusivity."""

import hashlib
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class ParticleDispersion:
    """Configure seeded particle dispersion on the accepted forcing sequence.

    Args:
        seed: Unsigned 64-bit integer selecting an independent ensemble.
        max_step: Maximum particle substep in seconds. Decrease this when
            resolving first contact with an open or depositing boundary.
    """

    seed: int
    max_step: float
    algorithm: ClassVar[str] = "prescribed-dispersion-v1"

    def __post_init__(self):
        """Validate the seed and duration before allocating particle arrays."""
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError(
                "supply an unsigned 64-bit particle dispersion seed"
            )
        if not np.isfinite(self.max_step) or self.max_step <= 0:
            raise ValueError("particle dispersion max_step must be positive")

    def configuration(self):
        """Return the restart contract for counter-based particle draws."""
        return dict(
            algorithm=self.algorithm, seed=self.seed, max_step=self.max_step
        )

    def keys(self, particle_ids):
        """Derive one stable key per identity, independent of storage order."""
        seed = self.seed.to_bytes(8, "little")
        return np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(
                        name.encode("utf-8"), digest_size=8, key=seed
                    ).digest(),
                    "little",
                )
                for name in particle_ids
            ),
            dtype=np.uint64,
        )

    def coefficients(self, grid, frame):
        """Return constant Cartesian K or full-interval jump fractions.

        Each shared face moves an equal fraction of the same physical volume
        in either direction. An absorbing exterior uses the distance from
        the cell center to the face; reflecting faces transfer zero mass.
        """
        active = np.asarray(grid.shape) > 1
        k = frame.diffusivity * active
        if self._uniform_cartesian(grid, frame, k, active):
            return k[0], None
        exchange = frame.mixing.copy()
        for side, mode in enumerate(grid.boundaries):
            axis = side // 2
            if not active[axis]:
                exchange[:, side] = 0
            elif mode in ("deposit", "open"):
                ref, area = grid.face_geometry(side)
                width = grid.spacing[axis]
                if axis == 2:
                    width = width * grid.jacobian(ref[:, 0])
                exterior = grid.neighbors[:, side] < 0
                conductance = 2 * k[:, axis] * abs(area[:, axis]) / width
                exchange[exterior, side] = (
                    frame.end - frame.start
                ) * conductance[exterior]
        return np.zeros(3), exchange / grid.volume[:, None]

    @staticmethod
    def _uniform_cartesian(grid, frame, k, active):
        """Check both cell samples and face conductances against constant K."""
        if grid.bed or not np.all(k == k[0]):
            return False
        for axis in np.flatnonzero(active):
            side = 2 * axis + 1
            internal = grid.neighbors[:, side] >= 0
            # A low exterior has no inward neighbor. Its interior counterpart
            # is covered by the positive face of the adjacent cell.
            expected = (
                (frame.end - frame.start)
                * k[0, axis]
                * grid.volume[internal]
                / grid.spacing[axis] ** 2
            )
            if not np.allclose(
                frame.mixing[internal, side],
                expected,
                rtol=16 * np.finfo(float).eps,
                atol=0,
            ):
                return False
        return True
