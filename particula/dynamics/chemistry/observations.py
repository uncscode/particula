"""Time-resolved observation likelihoods for kinetic mechanism discovery."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class ObservationSeries:
    """Finite observations and whitening factors at irregular sample times.

    Covariances have squared or cross-product measurement units. Their
    diagonals describe variances. Off-diagonal entries represent correlated
    channels at the same time. A missing observation is represented by NaN.
    Each observed block must be positive definite: omit one channel for an
    exact derived total. Distinct time points are treated as independent.
    """

    times: NDArray[np.float64]
    values: NDArray[np.float64]
    indices: tuple[NDArray[np.intp], ...]
    factors: tuple[NDArray[np.float64], ...]

    @classmethod
    def create(cls, times: ArrayLike, values: ArrayLike, covariance: ArrayLike):
        """Validate observations and shared or per-time covariance."""
        time = np.array(times, dtype=float, copy=True)
        data = np.array(values, dtype=float, copy=True)
        if time.ndim != 1 or not time.size or not np.isfinite(time).all():
            raise ValueError("Observation times must be a finite vector")
        if time[0] < 0 or np.any(np.diff(time) <= 0):
            raise ValueError(
                "Observation times must increase from zero or later"
            )
        if data.ndim != 2 or data.shape[0] != time.size or np.isinf(data).any():
            raise ValueError("Expected time by channel observations")
        if np.isnan(data).all():
            raise ValueError("At least one observation is required")
        count = data.shape[1]
        cov = np.asarray(covariance, dtype=float)
        if cov.shape == (count, count):
            cov = np.broadcast_to(cov, (time.size, count, count))
        if cov.shape != (time.size, count, count):
            raise ValueError("Expected channel covariance at each time")
        indices = tuple(np.flatnonzero(np.isfinite(row)) for row in data)
        factors = tuple(
            _factor(matrix[np.ix_(idx, idx)])
            for matrix, idx in zip(cov, indices, strict=True)
        )
        for array in (time, data, *indices, *factors):
            array.setflags(write=False)
        return cls(time, data, indices, factors)

    def loss(self, predictions: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Evaluate the Gaussian loss and its prediction gradient."""
        predicted = np.asarray(predictions, dtype=float)
        if predicted.shape != self.values.shape:
            raise ValueError("Prediction and observation shapes differ")
        gradient = np.zeros_like(predicted)
        total = 0.0
        for t, (idx, lower) in enumerate(
            zip(self.indices, self.factors, strict=True)
        ):
            if not idx.size:
                continue
            delta = predicted[t, idx] - self.values[t, idx]
            white = solve_triangular(lower, delta, lower=True)
            gradient[t, idx] = solve_triangular(lower.T, white)
            total += 0.5 * (white @ white + idx.size * np.log(2 * np.pi))
            total += np.log(lower.diagonal()).sum()
        return float(total), gradient


def _factor(covariance):
    """Check the observed covariance before using its lower triangle."""
    if not np.isfinite(covariance).all() or not np.array_equal(
        covariance, covariance.T
    ):
        raise ValueError("Observed covariance must be finite and symmetric")
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Observed covariance must be positive definite"
        ) from exc
