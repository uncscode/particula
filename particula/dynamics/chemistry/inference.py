"""Fit mechanism candidates to experiments with shared or local parameters."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from particula.dynamics.chemistry.discovery import Network, SearchResult

Vector = NDArray[np.float64]
Loss = Callable[[Vector, bool], tuple[float, Vector | None]]


@dataclass(frozen=True)
class ExperimentLoss:
    """Experiment objective and its mapping into the global parameter vector.

    Reusing an index across experiments shares the corresponding physical
    parameter. Using a distinct index makes it experiment-specific. The
    objective receives local physical values and a gradient request.
    """

    parameter_indices: tuple[int, ...]
    evaluate: Loss


@dataclass(frozen=True)
class FitResult:
    """Physical parameters, loss and diagnostics for each start."""

    parameters: Vector
    loss: float
    converged: tuple[bool, ...]
    evaluations: tuple[int, ...]


def _parameter_domain(bounds, starts):
    limits = np.asarray(bounds, dtype=float)
    starts_array = np.asarray(starts, dtype=float)
    if (
        limits.ndim != 2
        or limits.shape[1] != 2
        or not len(limits)
        or not np.isfinite(limits).all()
        or (limits[:, 1] < limits[:, 0]).any()
    ):
        raise ValueError("Expected finite ordered bounds")
    lower = limits[:, 0]
    width = limits[:, 1] - lower
    if (
        starts_array.ndim != 2
        or starts_array.shape[1] != len(lower)
        or not len(starts_array)
        or not np.isfinite(starts_array).all()
        or (starts_array < lower).any()
        or (starts_array > limits[:, 1]).any()
    ):
        raise ValueError("Expected one or more feasible starts")
    return lower, width, starts_array


def _minimize(objective, initial, gradient, max_iterations, max_evaluations):
    """Enforce the same objective-call ceiling for AD and finite differences."""
    best = OptimizeResult(x=initial.copy(), fun=np.inf, nfev=0, success=False)

    class BudgetReachedError(Exception):
        """Signal exhaustion before the next objective or perturbation."""

    def counted(x):
        if best.nfev >= max_evaluations:
            raise BudgetReachedError
        value = objective(x)
        best.nfev += 1
        loss = value[0] if gradient else value
        if loss < best.fun:
            best.x, best.fun = x.copy(), loss
        return value

    try:
        return minimize(
            counted,
            initial,
            method="L-BFGS-B",
            jac=True if gradient else "3-point",
            bounds=[(0.0, 1.0)] * len(initial),
            options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": 1e-8},
        )
    except BudgetReachedError:
        return best


def fit_experiments(
    experiments: Sequence[ExperimentLoss],
    bounds: Sequence[tuple[float, float]],
    starts: Sequence[Vector],
    *,
    gradient: bool = True,
    max_iterations: int = 100,
    max_evaluations: int = 1000,
) -> FitResult:
    """Fit finite box-bounded parameters in normalized linear coordinates.

    Zero-valued rates are feasible when the supplied lower bound is zero.
    Set the bounds equal to fix a parameter for a rate profile or channel
    deletion control. Finite-difference fits use the same bounds and starts.
    """
    lower, width, starts_array = _parameter_domain(bounds, starts)
    active = width > 0
    if not experiments or min(max_iterations, max_evaluations) < 1:
        raise ValueError(
            "Experiments and a positive iteration budget are required"
        )
    indices = [
        np.asarray(e.parameter_indices, dtype=np.intp) for e in experiments
    ]
    if any((i < 0).any() or (i >= len(lower)).any() for i in indices):
        raise ValueError("Experiment parameter index outside global bounds")

    def objective(x):
        params = lower.copy()
        params[active] += x * width[active]
        derivative = np.zeros_like(lower)
        total = 0.0
        for experiment, idx in zip(experiments, indices, strict=True):
            loss, grad = experiment.evaluate(params[idx], gradient)
            if not np.isfinite(loss):
                raise ValueError("Nonfinite experiment loss")
            total += loss
            if gradient:
                if np.shape(grad) != idx.shape or not np.isfinite(grad).all():
                    raise ValueError("Invalid experiment gradient")
                np.add.at(derivative, idx, grad)
        return (
            (total, derivative[active] * width[active]) if gradient else total
        )

    if not active.any():
        val = objective(np.empty(0))
        return FitResult(
            lower.copy(), float(val[0] if gradient else val), (True,), (1,)
        )
    fits = [
        _minimize(
            objective,
            (x[active] - lower[active]) / width[active],
            gradient,
            max_iterations,
            max_evaluations,
        )
        for x in starts_array
    ]
    best = min(fits, key=lambda r: r.fun)
    params = lower.copy()
    params[active] += width[active] * best.x
    return FitResult(
        params,
        float(best.fun),
        tuple(bool(r.success) for r in fits),
        tuple(int(r.nfev) for r in fits),
    )


@dataclass(frozen=True)
class CandidateFit:
    """Calibration and held-out experiments for one candidate."""

    experiments: Sequence[ExperimentLoss]
    bounds: Sequence[tuple[float, float]]
    starts: Sequence[Vector]
    held_out: Sequence[ExperimentLoss] = ()


@dataclass(frozen=True)
class RankedNetwork:
    """Mechanism and fit, with score gap measured from the best candidate."""

    network: Network
    fit: FitResult
    score: float
    delta: float
    competitive: bool


def fit_networks(
    search: SearchResult,
    prepare: Callable[[Network], CandidateFit],
    *,
    criterion: str,
    tolerance: float,
    gradient=True,
    max_iterations=100,
    max_evaluations=1000,
) -> tuple[RankedNetwork, ...]:
    """Rank every enumerated candidate using matched fitting budgets.

    For 'aic', each fit objective must be a negative log likelihood for the
    same observations and noise model. Count all free fitted parameters.
    For 'held_out', score untouched experiments at training estimates.
    The tolerance is in the selected score's units. All candidates within
    it remain competitive, including simpler parents. Search completeness
    and its stopping reason remain available on the supplied SearchResult.
    """
    if (
        criterion not in ("aic", "held_out")
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Invalid criterion or score tolerance")
    rows = []
    starts_count = None
    for network in search.networks:
        trial = prepare(network)
        if starts_count is None:
            starts_count = len(trial.starts)
        if len(trial.starts) != starts_count:
            raise ValueError("Candidates require the same number of starts")
        fit = fit_experiments(
            trial.experiments,
            trial.bounds,
            trial.starts,
            gradient=gradient,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
        )
        if criterion == "aic":
            score = 2 * (fit.loss + sum(b > a for a, b in trial.bounds))
        else:
            if not trial.held_out:
                raise ValueError(
                    "Held-out ranking requires evaluation experiments"
                )
            score = sum(
                exp.evaluate(
                    fit.parameters[list(exp.parameter_indices)], False
                )[0]
                for exp in trial.held_out
            )
        if not np.isfinite(score):
            raise ValueError("Nonfinite mechanism score")
        rows.append((float(score), network, fit))
    ordered = sorted(rows, key=lambda row: (row[0], len(row[1]), row[1]))
    return tuple(
        RankedNetwork(
            net,
            fit,
            score,
            score - ordered[0][0],
            score - ordered[0][0] <= tolerance,
        )
        for score, net, fit in ordered
    )
