"""Jointly vary AIOMFAC small-mode composition and two uptake probabilities."""

import numpy as np
import pandas as pd
import warp as wp
from field_fit import HERE, design_table
from particula.dynamics.chemistry.inference import (
    ExperimentLoss,
    fit_experiments,
)


@wp.kernel
def interpolated_loss(
    p: wp.array(dtype=wp.float64),
    low: wp.array2d(dtype=wp.float64),
    high: wp.array2d(dtype=wp.float64),
    y: wp.array(dtype=wp.float64),
    lx: wp.float64,
    hx: wp.float64,
    total: wp.array(dtype=wp.float64),
):
    """Accumulate log residuals at interpolated composition weights."""
    i = wp.tid()
    t = (p[2] - lx) / (hx - lx)
    a = low[i, 0] + t * (high[i, 0] - low[i, 0])
    b = low[i, 1] + t * (high[i, 1] - low[i, 1])
    predicted = a * wp.exp(p[0]) + b * wp.exp(p[1])
    resid = wp.log(predicted / y[i]) / wp.log(wp.float64(10.0))
    wp.atomic_add(total, 0, resid * resid / wp.float64(y.shape[0]))


class LocalModel:
    """Differentiate the tracer-bin operator between adjacent AIOMFAC nodes."""

    def __init__(self, low, high, obs, x0, x1):
        """Prepare the adjacent compositions and observed tracer marks."""
        self.arr = [
            wp.array(np.ascontiguousarray(x), dtype=wp.float64, device="cpu")
            for x in (low, high, obs)
        ]
        self.x0, self.x1 = x0, x1
        self.n = len(obs)

    def __call__(self, parameters, gradient):
        """Evaluate loss and optional gradient inside one grid interval."""
        p = wp.array(
            parameters, requires_grad=True, dtype=wp.float64, device="cpu"
        )
        objective = wp.zeros(
            1, requires_grad=True, dtype=wp.float64, device="cpu"
        )
        with wp.Tape() as tape:
            wp.launch(
                interpolated_loss,
                dim=self.n,
                inputs=[p, *self.arr, self.x0, self.x1, objective],
                device="cpu",
            )
        if gradient:
            tape.backward(objective)
        return objective.numpy().item(), p.grad.numpy() if gradient else None


def main():
    """Fit on five nodes and check the four intermediate compositions."""
    grid = pd.read_csv(HERE / "composition-grid.csv")
    obs = pd.read_csv(HERE / "evidence/decker-tracer-marks.csv")
    y = obs.gamma_marker.to_numpy()
    tracer = pd.read_csv(HERE.parent / "flight-windows.csv")[
        ["flight", "start_s", "n2o_ppbv"]
    ]
    grid = grid.merge(tracer, on=["flight", "start_s"], validate="many_to_one")
    grid["wet_total_area"] = grid.wet_small_area + grid.wet_other_area
    levels = np.sort(grid.f_org.unique())
    np.testing.assert_allclose(levels, np.linspace(0.25, 0.75, 9))
    results = []
    inactive = []
    for basis in ("wet", "dry"):
        designs = {}
        areas = {}
        for x, g in grid.groupby("f_org"):
            designs[x] = design_table(g, obs, basis)[["a", "b"]].to_numpy()
            # Closed-form scale for squared log error with the small mode off.
            weight = designs[x][:, 1]
            scale = np.exp(np.mean(np.log(y / weight)))
            inactive.append(
                dict(
                    basis=basis,
                    f_org=x,
                    gamma_small=0,
                    gamma_other=scale,
                    rmse_dex=np.sqrt(
                        np.mean(np.log10(scale * weight / y) ** 2)
                    ),
                )
            )
            areas[x] = g.sort_values(["flight", "start_s"])[
                ["wet_small_area", "wet_other_area"]
            ].to_numpy()
        for x0, mid, x1 in zip(
            levels[0:-1:2], levels[1::2], levels[2::2], strict=True
        ):
            model = LocalModel(designs[x0], designs[x1], y, x0, x1)
            p0 = np.array([np.log(0.008), np.log(0.025), mid])
            direction = np.array([0.6, -0.8, 0.1])
            grad = model(p0, True)[1] @ direction
            fd = [
                (
                    model(p0 + h * direction, False)[0]
                    - model(p0 - h * direction, False)[0]
                )
                / (2 * h)
                for h in [1e-3, 1e-4, 1e-5]
            ]
            np.testing.assert_allclose(fd, grad, rtol=1e-6, atol=1e-10)
            fit = fit_experiments(
                [ExperimentLoss((0, 1, 2), model)],
                [(-13.82, 0.0), (-13.82, 0.0), (x0, x1)],
                np.array(
                    [
                        p0,
                        [np.log(0.001), np.log(0.04), x1],
                        [np.log(0.02), np.log(0.003), x0],
                    ]
                ),
            )
            if not any(fit.converged):
                raise RuntimeError("Joint fit failed")
            params = fit.parameters
            gamma = np.exp(params[:2])
            direct_error = (
                designs[mid] - (designs[x0] + designs[x1]) / 2
            ) @ gamma
            midarea = (areas[x0] + areas[x1]) / 2
            results.append(
                dict(
                    basis=basis,
                    f_low=x0,
                    f_high=x1,
                    fitted_organic_fraction=params[2],
                    gamma_small=gamma[0],
                    gamma_other=gamma[1],
                    rmse_dex=np.sqrt(fit.loss),
                    check_f_org=mid,
                    maximum_midpoint_area_relative_error=np.max(
                        abs(midarea / areas[mid] - 1)
                    ),
                    max_midpoint_gamma_relative_error=np.max(
                        abs(direct_error / (designs[mid] @ gamma))
                    ),
                    max_ad_direction_absolute_error=np.max(
                        abs(np.array(fd) - grad)
                    ),
                )
            )
    pd.DataFrame(inactive).to_csv(HERE / "inactive-mode-check.csv", index=False)
    pd.DataFrame(results).to_csv(HERE / "joint-fit.csv", index=False)


if __name__ == "__main__":
    main()
