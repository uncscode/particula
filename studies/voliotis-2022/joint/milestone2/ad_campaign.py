"""CUDA forward-mode derivatives and matched refits for Manchester scores.

The archive's Warp model supplies all experiment data and verifies the primal.
The CUDA kernel propagates a first-order dual number through the 31-pool bulk
replay, returning each run's two normalized RMSEs and their sensitivities to
its dose scale and any nominated channel rates. The run maximum is handled
as epigraph constraints, so the optimizer can cross a change of worst run.
"""

import ctypes
import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import warp as wp
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent


@functools.cache
def _kernel():
    """Build for the current CUDA device in a temporary directory."""
    device = wp.get_device("cuda:0")
    source = HERE / "ad_campaign.cu"
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("CUDA Toolkit nvcc required for campaign JVPs")
    with tempfile.TemporaryDirectory() as directory:
        built = Path(directory) / "campaign_jvp.so"
        subprocess.run(  # noqa: S603 - fixed source and compiler arguments
            [
                nvcc,
                "-std=c++17",
                "-O3",
                f"-arch=sm_{device.arch}",
                "-shared",
                "-Xcompiler=-fPIC",
                str(source),
                "-o",
                str(built),
            ],
            check=True,
        )
        kernel = ctypes.CDLL(str(built)).campaign_jvp
    kernel.argtypes = (
        [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 3
    )
    kernel.restype = ctypes.c_int
    return kernel


class CampaignDerivatives:
    """Return per-run mass/count sensitivities in physical units.

    Supply channel IDs for rates or use -2-j for shared parameter j; the
    first derivative column always holds the run-specific dose sensitivity.
    """

    def __init__(self, model, active, tune, sub=4, *, keep_trajectory=False):
        """Bind a fixed topology and the required derivative directions."""
        active_set = set(active)
        if len(active_set) != len(active) or not active_set <= set(
            range(model.n_reactions)
        ):
            raise ValueError("Invalid or repeated reaction index")
        if (
            sub < 1
            or int(sub) != sub
            or not {h for h in tune if h >= 0} <= active_set
            or any(h == -1 or h < -14 for h in tune)
        ):
            raise ValueError(
                "Expected a positive substep count and active fitted rates"
            )
        if model.vapor_loss_h != 1.0 or tuple(model.masses) != (
            184.0,
            140.0,
            118.0,
        ):
            raise ValueError(
                "Expected archived vapor loss and equivalent masses"
            )
        self.model = model
        self.active = list(active)
        self.tune = list(tune)
        self.sub = int(sub)
        self.kernel = _kernel()
        self.runs = len(model.runs)
        self.directions = len(tune) + 1
        self.ids = self._upload(active, wp.int32)
        self.seeds = self._upload([-1, *tune], wp.int32)
        self.minute_tangents = (
            wp.empty(
                (self.runs, self.directions, 300, 2),
                dtype=wp.float64,
                device="cuda:0",
            )
            if keep_trajectory
            else None
        )
        self.output = wp.empty(
            (self.runs, self.directions, 4), dtype=wp.float64, device="cuda:0"
        )

    @staticmethod
    def _upload(data, dtype=wp.float64):
        return wp.array(
            np.ascontiguousarray(data), dtype=dtype, device="cuda:0"
        )

    def evaluate(self, p, rates, doses):
        """Return RMSE, local-dose sensitivity and shared sensitivities."""
        p, rates, doses = (
            np.asarray(v, dtype=np.float64) for v in (p, rates, doses)
        )
        if (
            p.shape != (16,)
            or rates.shape != (self.model.n_reactions,)
            or doses.shape != (self.runs,)
            or not all(np.isfinite(v).all() for v in (p, rates, doses))
            or (rates < 0).any()
            or (doses <= 0).any()
        ):
            raise ValueError("Invalid campaign parameter dimensions or values")
        params = [self._upload(v) for v in (p, rates, doses)]
        stream = wp.get_stream("cuda:0")
        with wp.ScopedDevice("cuda:0"):
            args = [
                *self.model.data,
                self.model.channels,
                self.ids,
                *params,
                self.seeds,
            ]
            status = self.kernel(
                *(a.ptr for a in args),
                self.runs,
                self.directions,
                len(self.active),
                self.sub,
                self.output.ptr,
                self.minute_tangents.ptr
                if self.minute_tangents is not None
                else None,
                stream.cuda_stream,
            )
            if status:
                raise RuntimeError(
                    f"CUDA campaign launch returned status {status}"
                )
            data = self.output.numpy()
        if not np.isfinite(data).all():
            bad = np.argwhere(~np.isfinite(data))[0]
            raise ArithmeticError(
                f"Nonfinite run, direction, output index: {bad.tolist()}"
            )
        return (
            data[:, 0, :2],
            data[:, 0, 2:],
            np.transpose(data[:, 1:, 2:], (0, 2, 1)),
        )


def fit_rates_and_doses(
    model,
    p,
    rates,
    active,
    doses,
    tune,
    *,
    sub=4,
    max_evaluations=30,
    tolerance=1e-8,
    tune_physics=(),
    both_orders=False,
):
    """Refit physical parameters, rates and doses under a call ceiling.

    Prices each joint trajectory/Jacobian call once; the best evaluated feasible
    point is retained even when the optimizer exhausts the budget.
    """
    if max_evaluations < 1:
        raise ValueError("Expected a positive evaluation budget")
    source_slots = np.array(tune_physics, dtype=int)
    rate_slots = np.array(tune, dtype=int)
    # The same physical bounds as the campaign's archived minimax refit.
    lower_shared = np.r_[[1 / 12] * 3, [0.0] * 6, 2.818, 0, 0, 0]
    upper_shared = np.r_[[4.0] * 3, [2.0] * 6, 16.5515, 0, 0, 100]
    p = np.asarray(p, dtype=float)
    rates = np.asarray(rates, dtype=float)
    doses = np.asarray(doses, dtype=float)
    rows = model.channels.numpy()
    caps = np.where(rows[rate_slots, 1] >= 0, 25.0, 5.0)
    lower = np.r_[lower_shared[source_slots], np.zeros(len(tune))]
    span = np.r_[upper_shared[source_slots] - lower_shared[source_slots], caps]
    shared = np.r_[p[source_slots], rates[rate_slots]]
    if (
        (span <= 0).any()
        or (shared < lower - 1e-12).any()
        or (shared > lower + span + 1e-12).any()
        or (doses < 0.85).any()
        or (doses > 1.15).any()
    ):
        raise ValueError("Initial parameters violate campaign bounds")
    directions = [*(-2 - source_slots), *tune]
    orders = [active, list(reversed(active))] if both_orders else [active]
    engines = [
        CampaignDerivatives(model, order, directions, sub) for order in orders
    ]
    n = len(directions)
    start = np.r_[np.clip((shared - lower) / span, 0, 1), (doses - 0.85) / 0.3]
    history = []

    class BudgetReachedError(Exception):
        """Interrupt before an unbudgeted trajectory/Jacobian evaluation."""

    @functools.lru_cache(maxsize=4)
    def evaluate(key):
        if len(history) >= max_evaluations:
            raise BudgetReachedError
        x = np.array(key)
        par, k = p.copy(), rates.copy()
        physical = lower + span * x[:n]
        par[source_slots] = physical[: len(source_slots)]
        k[rate_slots] = physical[len(source_slots) :]
        d = 0.85 + 0.3 * x[n:]
        errors, derivatives = [], []
        for engine in engines:
            err, by_dose, by_rate = engine.evaluate(par, k, d)
            derivative = np.zeros((len(d), 2, len(x)))
            derivative[:, :, :n] = by_rate * span
            for r in range(len(d)):
                derivative[r, :, n + r] = by_dose[r] * 0.3
            errors.append(err.ravel())
            derivatives.append(derivative.reshape(-1, len(x)))
        errors = np.concatenate(errors)
        derivative = np.vstack(derivatives)
        history.append((float(errors.max()), par, k, d))
        return errors.ravel(), derivative.reshape(-1, len(x))

    value = evaluate(tuple(start))[0].max()
    init = np.r_[start, value]
    obj_grad = np.zeros_like(init)
    obj_grad[-1] = 1.0

    def jacobian(x):
        deriv = evaluate(tuple(x[:-1]))[1]
        return np.column_stack((-deriv, np.ones(len(deriv))))

    try:
        fit = minimize(
            lambda x: x[-1],
            init,
            jac=lambda _: obj_grad,
            bounds=[(0, 1)] * len(start) + [(0, None)],
            constraints=[
                dict(
                    type="ineq",
                    fun=lambda x: x[-1] - evaluate(tuple(x[:-1]))[0],
                    jac=jacobian,
                )
            ],
            method="SLSQP",
            options=dict(maxiter=max_evaluations, ftol=tolerance),
        )
        status = str(fit.message)
    except BudgetReachedError:
        status = "Evaluation budget exhausted"
    score, par, k, d = min(history, key=lambda entry: entry[0])
    return dict(
        score=score,
        initial_score=float(value),
        x=np.r_[par, k].tolist(),
        dose=d.tolist(),
        evaluations=len(history),
        status=status,
    )
