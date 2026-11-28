"""Fit tracer-bin uptake marks and calculate window-specific loss rates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from particula.dynamics.chemistry.inference import (
    ExperimentLoss,
    fit_experiments,
)
from particula.util.constants import GAS_CONSTANT

HERE = Path(__file__).resolve().parent
MASS = 0.1080104
FLIGHTS = (20230308, 20230311, 20230315)


@wp.kernel
def log_loss(
    p: wp.array(dtype=wp.float64),
    a: wp.array(dtype=wp.float64),
    b: wp.array(dtype=wp.float64),
    obs: wp.array(dtype=wp.float64),
    n: int,
    total: wp.array(dtype=wp.float64),
):
    """Accumulate squared log residuals across tracer bins."""
    i = wp.tid()
    calc = a[i] * wp.exp(p[0]) + b[i] * wp.exp(p[1])
    err = wp.log(calc / obs[i]) / wp.log(wp.float64(10.0))
    wp.atomic_add(total, 0, err * err / wp.float64(n))


class Loss:
    """Differentiate an equal-weight log error in two uptake probabilities."""

    def __init__(self, a, b, obs):
        """Copy the two area weights and reference marks to the device."""
        self.cols = [
            wp.array(np.ascontiguousarray(x), dtype=wp.float64, device="cpu")
            for x in (a, b, obs)
        ]
        self.n = len(obs)

    def __call__(self, values, gradient=True):
        """Evaluate the loss and optionally its reverse-mode derivative."""
        p = wp.array(values, dtype=wp.float64, requires_grad=True, device="cpu")
        loss = wp.zeros(1, dtype=wp.float64, requires_grad=True, device="cpu")
        with wp.Tape() as tape:
            wp.launch(
                log_loss,
                dim=self.n,
                inputs=[p, *self.cols, self.n, loss],
                device="cpu",
            )
        if gradient:
            tape.backward(loss)
        return loss.numpy().item(), p.grad.numpy() if gradient else None


def fit(design, target, model, fixed=None):
    """Fit shared, two-class, or fixed-small-gamma models with Particula."""
    same = model == "shared"
    bounds = [(-13.82, 0.0)] * (1 if same else 2)
    starts = (
        np.log([[0.01], [0.06]])
        if same
        else np.log([[0.002, 0.06], [0.02, 0.01], [0.04, 0.08]])
    )
    if fixed is not None:
        v = np.log(fixed)
        bounds[0] = (v, v)
        starts[:, 0] = v
    loss = Loss(*design.T, target)
    objective = ExperimentLoss((0, 0) if same else (0, 1), loss)
    result = fit_experiments(
        [objective], bounds, starts, gradient=True, max_iterations=150
    )
    if not any(result.converged):
        raise RuntimeError("No successful bounded field fit")
    p = np.repeat(result.parameters, 2) if same else result.parameters
    return np.exp(p), np.sqrt(result.loss)


def score(pred, obs):
    """Return root mean square difference in log10 uptake probability."""
    return np.sqrt(np.mean(np.log10(pred / obs) ** 2))


def design_table(frame, marks, basis):
    """Group the observation operator over the same three flight dates."""
    df = frame[frame.flight.isin(FLIGHTS)].copy()
    df["n2o_lo"] = (df.n2o_ppbv // 5 * 5).astype(int)
    denom = df["wet_total_area" if basis == "wet" else "dry_total_area"]
    df["a"] = df.wet_small_area / denom
    df["b"] = df.wet_other_area / denom
    bins = df.groupby("n2o_lo").agg(
        a=("a", "mean"), b=("b", "mean"), n=("a", "size")
    )
    return marks.merge(bins, on="n2o_lo", validate="one_to_one")


def fit_table(table, case, basis):
    """Evaluate the calibration, fixed alternate bins and leave-one-bin-out."""
    design = table[["a", "b"]].to_numpy()
    y = table.gamma_marker.to_numpy()
    train = (table.n2o_lo.to_numpy() % 10) == 0
    summaries, rows = [], []
    for model in ("shared", "two-class"):
        theta, err = fit(design, y, model)
        trained, _ = fit(design[train], y[train], model)
        loo = np.array(
            [
                design[i]
                @ fit(np.delete(design, i, axis=0), np.delete(y, i), model)[0]
                for i in range(len(y))
            ]
        )
        summaries.append(
            dict(
                case=case,
                basis=basis,
                model=model,
                bins=len(y),
                windows=int(table.n.sum()),
                gamma_small=theta[0],
                gamma_other=theta[1],
                train_all_RMSE_dex=err,
                alternate_test_RMSE_dex=score(
                    design[~train] @ trained, y[~train]
                ),
                loo_RMSE_dex=score(loo, y),
                matrix_condition=np.linalg.cond(design),
            )
        )
        for i, record in enumerate(table.itertuples()):
            rows.append(
                dict(
                    case=case,
                    basis=basis,
                    model=model,
                    n2o_lo=record.n2o_lo,
                    n_windows=record.n,
                    a=design[i, 0],
                    b=design[i, 1],
                    gamma_marker=y[i],
                    fitted=design[i] @ theta,
                    alternate=design[i] @ trained,
                    alternate_train=bool(train[i]),
                    loo=loo[i],
                )
            )
    return summaries, rows


def rates(frame, gamma):
    """Multiply gas collision frequencies by uptake probabilities."""
    v4 = np.sqrt(8 * GAS_CONSTANT * frame.temperature_k / (np.pi * MASS)) * 0.25
    small = v4 * frame.wet_small_area * gamma[0] * 1e-6
    other = v4 * frame.wet_other_area * gamma[1] * 1e-6
    cutoff = (
        v4
        * (
            frame.wet_small_below150_area * gamma[0]
            + frame.wet_other_below150_area * gamma[1]
        )
        * 1e-6
    )
    return pd.DataFrame(
        dict(
            flight=frame.flight,
            start_s=frame.start_s,
            k_small_s=small,
            k_other_s=other,
            k_total_s=small + other,
            k_below150_s=cutoff,
            lifetime_days=1 / (small + other) / 86400,
            loss_pptv_per_hour_per_pptv=3600 * (small + other),
            mode_loss_fraction=small / (small + other),
            cutoff_loss_fraction=cutoff / (small + other),
        )
    )


def save_optical_examples(ref, shared, theta):
    """Save matched sub-150 nm wet areas and loss shares for both fits."""
    keys = ["flight", "start_s"]
    optical = pd.read_csv(HERE.parent / "accuracy-window-diagnostics.csv")
    examples = optical.loc[optical.cutoff_in_accuracy_band, keys].merge(
        ref[keys + ["dry_below150_area", "wet_below150_area"]],
        on=keys,
        validate="one_to_one",
    )
    for label, gamma in [("shared", shared), ("two_class", theta)]:
        subset = rates(ref, gamma)[keys + ["cutoff_loss_fraction", "k_total_s"]]
        subset = subset.rename(
            columns={col: f"{label}_{col}" for col in subset if col not in keys}
        )
        examples = examples.merge(subset, on=keys, validate="one_to_one")
    examples.to_csv(HERE / "optical-examples.csv", index=False)


def main():
    """Save parameter profiles, transfer checks, rates and AD verification."""
    hydration = pd.read_csv(HERE / "hydration.csv")
    tracer = pd.read_csv(HERE.parent / "flight-windows.csv")[
        ["flight", "start_s", "n2o_ppbv"]
    ]
    source = hydration.merge(
        tracer, on=["flight", "start_s"], validate="many_to_one"
    )
    marks = pd.read_csv(HERE / "evidence/decker-tracer-marks.csv")
    fits, bins, scenarios, profiles = [], [], [], []
    ref = source[source.case == "aiomfac"]
    baseline = design_table(ref, marks, "wet")
    design = baseline[["a", "b"]].to_numpy()
    y = baseline.gamma_marker.to_numpy()
    theta, _ = fit(design, y, "two-class")
    loss = Loss(*design.T, y)
    x = np.log([0.01, 0.035])
    v = np.array([0.8, -0.6])
    ad = loss(x)[1] @ v
    fd = [
        (loss(x + h * v, False)[0] - loss(x - h * v, False)[0]) / (2 * h)
        for h in [1e-3, 1e-4, 1e-5]
    ]
    np.testing.assert_allclose(fd, ad, rtol=2e-6, atol=1e-9)
    for fixed in [
        1e-6,
        0.001,
        0.003,
        0.005,
        0.008,
        0.010,
        0.015,
        0.020,
        0.030,
        0.05,
        0.1,
    ]:
        params, err = fit(design, y, "two-class", fixed=fixed)
        profiles.append(
            dict(
                fixed_gamma_small=fixed,
                optimized_gamma_other=params[1],
                RMSE_dex=err,
            )
        )
    ratio_rows = []
    for small_to_other in [0.0, 0.25, 0.5, 1.0]:
        constrained = design.copy()
        constrained[:, 0] *= small_to_other
        common, rmse = fit(constrained, y, "shared")
        g = common * np.array([small_to_other, 1.0])
        r = rates(ref, g)
        ratio_rows.append(
            dict(
                small_to_other=small_to_other,
                gamma_small=g[0],
                gamma_other=g[1],
                RMSE_dex=rmse,
                median_rate_s=r.k_total_s.median(),
                median_mode_rate_share=r.mode_loss_fraction.median(),
            )
        )
    omitted = []
    for i, item in enumerate(baseline.itertuples()):
        g, err = fit(np.delete(design, i, axis=0), np.delete(y, i), "two-class")
        omitted.append(
            dict(
                excluded_n2o_lower=item.n2o_lo,
                gamma_small=g[0],
                gamma_other=g[1],
                retained_RMSE_dex=err,
                omitted_prediction=design[i] @ g,
                omitted_marker=y[i],
            )
        )
    pd.DataFrame(omitted).to_csv(HERE / "leave-one-out.csv", index=False)
    pd.DataFrame(ratio_rows).to_csv(
        HERE / "field-channel-ratios.csv", index=False
    )
    shared, _ = fit(design, y, "shared")
    save_optical_examples(ref, shared, theta)
    for case, frame in source.groupby("case"):
        for basis in ["wet", "dry"]:
            table = design_table(frame, marks, basis)
            if len(table) != len(marks):
                raise ValueError(f"Missing matched tracer bin for {case}")
            summary, rows = fit_table(table, case, basis)
            fits.extend(summary)
            bins.extend(rows)
        for model, gamma in [
            ("shared", shared),
            ("two-class", theta),
        ]:
            r = rates(frame, gamma)
            if case == "aiomfac":
                r.to_csv(HERE / f"rates-{model}.csv", index=False)
            scenarios.append(
                dict(
                    case=case,
                    model=model,
                    windows=len(frame),
                    median_wet_over_dry=np.median(
                        frame.wet_total_area / frame.dry_total_area
                    ),
                    median_rate_s=r.k_total_s.median(),
                    median_lifetime_days=r.lifetime_days.median(),
                    median_mode_area_share=np.median(
                        frame.wet_small_area / frame.wet_total_area
                    ),
                    median_mode_rate_share=r.mode_loss_fraction.median(),
                    median_below150_rate_share=r.cutoff_loss_fraction.median(),
                )
            )
    for name, rows in [
        ("field-fits", fits),
        ("tracer-bin-predictions", bins),
        ("area-rate-scenarios", scenarios),
        ("field-profile", profiles),
    ]:
        pd.DataFrame(rows).to_csv(HERE / f"{name}.csv", index=False)
    checks = dict(
        ad_directional=ad,
        finite_difference=fd,
        source="Decker 2024 preliminary poster",
        model=(
            "conditional on tabulated thermodynamics; "
            "Warp float64 CPU backward and bounded Particula fit"
        ),
    )
    (HERE / "field-checks.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
