"""Concrete deterministic ODE stepping, separate from runnable finite-step maps.

Rates have units of state per second. Callbacks receive detached read-only
arrays and must be pure: no RNG, observation, or external mutation. The caller
owns state scaling, physical admissibility and linear conservation invariants.
Bogacki-Shampine RK3(2) is for nonstiff rates; ROS-2/3 use the KPP coefficients:
https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html
These names are concrete implementation APIs, not package-root exports.
"""

from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import eye as sparse_eye
from scipy.sparse import issparse, sparray, spmatrix
from scipy.sparse.linalg import splu

Array = np.ndarray
Rate = Callable[[float, Array], Array]
Jacobian = Callable[[float, Array], Array | spmatrix | sparray]


class TrialDomainError(ValueError):
    """A proposed intermediate state is outside the model's physical domain."""


@dataclass(frozen=True)
class ODEProblem:
    """Declare pure continuous derivatives and optional stiff-method inputs.

    jacobian[i,j] is df_i/dy_j. ROS-2/3 require both jacobian and
    time_derivative (partial f/partial t at fixed state); autonomous problems
    explicitly supply zeros for the latter. admissible checks all internal
    stages and the accepted candidate, without changing either.
    """

    rate: Rate
    admissible: Callable[[Array], bool]
    jacobian: Jacobian | None = None
    time_derivative: Rate | None = None
    validate_candidate: Rate | None = None


@dataclass
class StepStatistics:
    """Count trial work separately from accepted steps."""

    accepted: int = 0
    rejected: int = 0
    domain_rejected: int = 0
    rate_calls: int = 0
    jacobian_calls: int = 0
    time_derivative_calls: int = 0
    factorizations: int = 0
    linear_solves: int = 0


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


class AdaptiveSolver:
    """Own accepted ODE state and advance one error-controlled step at a time.

    Errors use max(abs(error)/(atol + rtol*max(abs(old),abs(new)))). Scalars
    or per-component tolerances use the state units, with strictly positive
    atol. Domain rejection reduces the step without clipping inventories.
    A failed call preserves the last accepted state/time, but retains work
    counters. Earlier accepted steps are never rolled back. Output/restart
    orchestration observes this owner only between successful step() calls.
    """

    def __init__(
        self,
        problem,
        state,
        *,
        time=0.0,
        method="rk23",
        rtol=1e-5,
        atol=1e-12,
        first_step=1e-3,
        min_step=1e-14,
        max_step=1.0,
        max_trials=100,
    ):
        """Validate configuration and detach initial state before any rates."""
        if not isinstance(problem, ODEProblem):
            raise TypeError("problem must explicitly declare continuous rates")
        if method not in ("rk23", "ros2", "ros3"):
            raise ValueError("method must be rk23, ros2 or ros3")
        if method in ("ros2", "ros3") and (
            problem.jacobian is None or problem.time_derivative is None
        ):
            raise ValueError(
                f"{method} requires Jacobian and partial time derivative"
            )
        value = np.array(state, dtype=np.float64, copy=True)
        if value.ndim != 1 or not value.size:
            raise ValueError("state must be a nonempty one-dimensional array")
        self.problem = problem
        self.method = method
        self._check(value)
        self._state = value
        self._time = self._scalar(time, "time", positive=False)
        self.min_step = self._scalar(min_step, "min_step")
        self.max_step = self._scalar(max_step, "max_step")
        first_step = self._scalar(first_step, "first_step")
        if not self.min_step <= first_step <= self.max_step:
            raise ValueError("require min_step <= first_step <= max_step")
        if (
            isinstance(max_trials, bool)
            or not isinstance(max_trials, int)
            or max_trials < 1
        ):
            raise ValueError("max_trials must be a positive integer")
        self.max_trials = max_trials
        self.rtol = self._tolerance(rtol, value.shape, "rtol", False)
        self.atol = self._tolerance(atol, value.shape, "atol", True)
        self.next_step = first_step
        self.statistics = StepStatistics()
        self.last_error = None

    @staticmethod
    def _scalar(value, name, positive=True):
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError(f"{name} must be a finite scalar")
        value = float(value)
        if not np.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"{name} is outside its valid range")
        return value

    @staticmethod
    def _tolerance(value, shape, name, positive):
        values = np.asarray(value, dtype=np.float64)
        if values.shape not in ((), shape):
            raise ValueError(f"{name} must be scalar or match the state")
        if not np.all(np.isfinite(values)) or np.any(
            values <= 0 if positive else values < 0
        ):
            raise ValueError(f"{name} is outside its valid range")
        return _readonly(np.broadcast_to(values, shape))

    @property
    def state(self):
        """Return detached accepted state in the caller's declared units."""
        return self._state.copy()

    @property
    def time(self):
        """Return accepted absolute time in seconds."""
        return self._time

    def _check(self, state):
        if not np.all(np.isfinite(state)) or not self.problem.admissible(
            _readonly(state)
        ):
            raise TrialDomainError("trial state is outside the physical domain")

    def _rate(self, time, state):
        self._check(state)
        self.statistics.rate_calls += 1
        result = np.asarray(
            self.problem.rate(time, _readonly(state)), dtype=np.float64
        )
        if result.shape != state.shape:
            raise ValueError("rate must have exactly the state shape")
        if not np.all(np.isfinite(result)):
            raise TrialDomainError("trial derivative is nonfinite")
        return result.copy()

    def _rk23(self, t, y, h):
        k1 = self._rate(t, y)
        k2 = self._rate(t + h / 2, y + h * k1 / 2)
        k3 = self._rate(t + 3 * h / 4, y + 3 * h * k2 / 4)
        candidate = y + h * (2 * k1 / 9 + k2 / 3 + 4 * k3 / 9)
        k4 = self._rate(t + h, candidate)
        error = h * (-5 * k1 / 72 + k2 / 12 + k3 / 9 - k4 / 8)
        return candidate, error

    def _linearize(self, t, y, h, gamma):
        """Factor the Rosenbrock matrix once for all stages of one trial."""
        jacobian = self.problem.jacobian
        time_derivative = self.problem.time_derivative
        if jacobian is None or time_derivative is None:
            raise ValueError("Rosenbrock requires both derivative callbacks")
        rate = self._rate(t, y)
        self.statistics.jacobian_calls += 1
        raw_jac = cast(Any, jacobian(t, _readonly(y)))
        sparse = issparse(raw_jac)
        jac = (
            raw_jac.astype(np.float64).tocsc()
            if sparse
            else np.asarray(raw_jac, dtype=np.float64)
        )
        self.statistics.time_derivative_calls += 1
        partial = np.asarray(time_derivative(t, _readonly(y)), dtype=np.float64)
        if jac.shape != (y.size, y.size) or partial.shape != y.shape:
            raise ValueError("Jacobian or time derivative has incorrect shape")
        if not np.all(np.isfinite(jac.data if sparse else jac)) or not np.all(
            np.isfinite(partial)
        ):
            raise TrialDomainError("nonfinite linearization")
        if sparse:
            matrix = sparse_eye(y.size, format="csc") / (h * gamma) - jac
            factor = splu(matrix)
            raw_solve = factor.solve
        else:
            matrix = np.eye(y.size) / (h * gamma) - jac
            dense_factor = lu_factor(matrix)

            def raw_solve(rhs):
                return lu_solve(dense_factor, rhs)

        self.statistics.factorizations += 1

        def solve(rhs):
            result = raw_solve(rhs)
            self.statistics.linear_solves += 1
            # Pivoting can feed roundoff into zero-rate inventory coordinates.
            # Correct the residual with the same factors, without clipping
            # coordinates or treating small physical rates as zero.
            for _ in range(2):
                result += raw_solve(rhs - matrix @ result)
                self.statistics.linear_solves += 1
            return result

        return rate, partial, solve

    def _ros2(self, t, y, h):
        gamma = 1 + 1 / np.sqrt(2)
        rate, partial, solve = self._linearize(t, y, h, gamma)
        k1 = solve(rate + h * gamma * partial)
        rate2 = self._rate(t + h, y + k1 / gamma)
        k2 = solve(rate2 - 2 * k1 / (gamma * h) - h * gamma * partial)
        candidate = y + (3 * k1 + k2) / (2 * gamma)
        self._check(candidate)
        return candidate, (k1 + k2) / (2 * gamma)

    def _ros3(self, t, y, h):
        """Take a KPP ROS-3 trial with an embedded second-order estimate.

        Three stage solves share one factorization and two rate evaluations.
        Each stage includes two residual-correction solves, counted in stats.
        Coefficients: KineticPreProcessor/KPP int/rosenbrock.f90, Ros3,
        Git blob baf1ef64538a5128db5e1154ac7fd1b39c41f665. The Jacobian
        convention here is df_i/dy_j, without KPP's storage transpose.
        """
        gamma = 0.43586652150845899941601945119356
        rate, partial, solve = self._linearize(t, y, h, gamma)
        k1 = solve(rate + h * gamma * partial)
        rate2 = self._rate(t + gamma * h, y + k1)
        k2 = solve(
            rate2
            - 1.0156171083877702091975600115545 * k1 / h
            + h * 0.24291996454816804366592249683314 * partial
        )
        k3 = solve(
            rate2
            + (
                4.0759956452537699824805835358067 * k1
                + 9.2076794298330791242156818474003 * k2
            )
            / h
            + h * 2.1851380027664058511513169485832 * partial
        )
        candidate = (
            y
            + k1
            + 6.1697947043828245592553615689730 * k2
            - 0.42772256543218573326238373806514 * k3
        )
        self._check(candidate)
        error = (
            0.5 * k1
            - 2.9079558716805469821718236208017 * k2
            + 0.22354069897811569627360909276199 * k3
        )
        return candidate, error

    def step(self, bound):
        """Accept one step no later than bound [s]; return its accepted time.

        At an already reached bound this is a no-op. Only explicit domain
        failures are retried; configuration errors and callback bugs propagate.
        Repeated rejection or unrepresentable time advance raises RuntimeError.
        """
        bound = self._scalar(bound, "bound", positive=False)
        if bound < self.time:
            raise ValueError("bound precedes accepted time")
        if bound == self.time:
            return self.time
        h = min(self.next_step, self.max_step, bound - self.time)
        floor = min(self.min_step, bound - self.time)
        last_domain_error = None
        for _ in range(self.max_trials):
            if h < floor or self.time + h == self.time:
                if last_domain_error is not None:
                    raise RuntimeError(
                        f"{self.method} step below min_step: "
                        f"{last_domain_error}"
                    ) from last_domain_error
                raise RuntimeError("accuracy requires a step below min_step")
            domain_failure = False
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    trial = {
                        "rk23": self._rk23,
                        "ros2": self._ros2,
                        "ros3": self._ros3,
                    }[self.method]
                    candidate, error = trial(self.time, self._state, h)
                    if (
                        self.method != "rk23"
                        and self.problem.validate_candidate is not None
                    ):
                        self.problem.validate_candidate(
                            self.time + h, _readonly(candidate)
                        )
                    last_domain_error = None
                    scale = self.atol + self.rtol * np.maximum(
                        np.abs(self._state), np.abs(candidate)
                    )
                    norm = float(np.max(np.abs(error) / scale))
            except (TrialDomainError, FloatingPointError) as error:
                last_domain_error = error
                domain_failure = True
                norm = np.inf
            if norm <= 1:
                self._state = candidate
                self._time = bound if h == bound - self.time else self.time + h
                self.statistics.accepted += 1
                self.last_error = norm
                exponent = -1 / (2 if self.method == "ros2" else 3)
                factor = (
                    5.0
                    if norm == 0
                    else min(5.0, max(0.2, 0.9 * norm**exponent))
                )
                self.next_step = min(
                    self.max_step, max(self.min_step, h * factor)
                )
                return self.time
            self.statistics.rejected += 1
            self.statistics.domain_rejected += int(domain_failure)
            exponent = -1 / (2 if self.method == "ros2" else 3)
            h *= (
                0.5
                if not np.isfinite(norm)
                else max(0.1, min(0.5, 0.9 * norm**exponent))
            )
        if last_domain_error is not None:
            raise RuntimeError(
                f"{self.method} maximum trial count exceeded: "
                f"{last_domain_error}"
            ) from last_domain_error
        raise RuntimeError(
            "maximum trial count exceeded without accepting a step"
        )
