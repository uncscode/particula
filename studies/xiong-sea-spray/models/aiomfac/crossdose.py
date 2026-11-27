# %% Jointly constrain candidate networks with the 200 ppm observations and early 20 ppm retrieval.
import json
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from fit import ROOT, PARAMS, VOL, config, observations, training_weights, make_model
from particula.dynamics.chemistry.inference import ExperimentLoss, fit_experiments


def main():
    out = ROOT / "crossdose"
    out.mkdir(exist_ok=True)
    low = pd.read_csv(ROOT / "low_so2/s20.csv")
    idx = jnp.array((2 * low.time_h).astype(int).to_numpy() - 1)
    target = jnp.array(low[["sulfate_mol_per_L", "log_c0_c"]].to_numpy())
    data, ic = observations()
    wg, wi = training_weights(data, ic, "primary")
    e = pd.read_csv(ROOT.parent / "xiong/endmembers.csv").query("rh_pct==80")
    c0 = float(1000 / (VOL[0] * e.nacl_gf.iloc[0] ** 3))
    _, predict, residual = make_model(24)

    def outputs(p, flags):
        x, g, _ = predict(p[:-1], flags, p[-1])
        nacl, exposed_nacl, bisulfite, bisulfate, neutral_salt = x[idx, 1].T
        dry = (
            (nacl + exposed_nacl + (1 - p[8]) * bisulfite) * VOL[0]
            + (bisulfate + p[8] * bisulfite) * VOL[1]
            + 2 * neutral_salt * VOL[2]
        )
        wet = dry * g[idx, 1, 4] ** 3 / 1000
        return jnp.stack(
            [
                (bisulfate + neutral_salt) / wet,
                jnp.log(c0 * wet / jnp.maximum(nacl + exposed_nacl, 1e-12)),
            ],
            axis=1,
        )

    def loss(p, flags, wlow):
        dg, di = residual(p[:-1], flags)
        z = (outputs(p, flags) - target) / jnp.array([0.12, 0.12])
        return 0.5 * (wg @ (dg**2) + wi @ (di**2) + jnp.sum(wlow * z[:3] ** 2))

    deriv = jax.jit(jax.value_and_grad(loss))
    scores = []
    rows = []
    checks = []
    pars = pd.concat(
        [pd.read_csv(ROOT / tag / "scores.csv") for tag in ("primary", "retention")]
    ).set_index("network")
    for setting in ["early_joint", "early_sulfate", "linear_joint"]:
        wt = (
            np.array([0.5, 0.5]) if setting != "early_sulfate" else np.array([1.0, 0.0])
        )
        for name in [
            "direct",
            "siv_lag",
            "access_delay",
            "neutral",
            "parallel_neutral",
            "retention_neutral",
        ]:
            flags, bounds = config(name)
            bounds = np.vstack(
                [bounds, [0.10, 0.10] if setting == "linear_joint" else [0.025, 1.0]]
            )
            baseline = pars.loc[name, PARAMS].to_numpy(dtype=float)
            seeds = [np.r_[baseline, v] for v in (0.13, 0.35)]
            seeds.append(
                np.r_[[1.5, 0.8, 0.8, 2, 0.5, 0.3, 0.25, -0.5, 0.5, 0.0], 0.22]
            )
            starts = [np.clip(p, *bounds.T) for p in seeds]

            def value(p, gradient):
                value, grad = deriv(p, flags, wt)
                return float(value), np.asarray(grad)

            fit = fit_experiments(
                [ExperimentLoss(tuple(range(len(bounds))), value)],
                bounds,
                starts,
                max_iterations=220,
                max_evaluations=440,
            )
            p = fit.parameters
            ans = np.array(outputs(p, flags))
            dg, di = map(np.array, residual(p[:-1], flags))
            errs = ans - np.array(target)
            h = 1e-5
            probe = 0.75 * starts[0] + 0.25 * bounds.mean(axis=1)
            v = (bounds[:, 1] > bounds[:, 0]) * np.cos(np.arange(len(p)) + 1)
            ad = value(probe, True)[1] @ v
            fd = (value(probe + h * v, False)[0] - value(probe - h * v, False)[0]) / (
                2 * h
            )
            checks.append(
                dict(
                    network=name,
                    setting=setting,
                    converged=fit.converged,
                    evaluations=fit.evaluations,
                    relative_gradient_error=float(
                        abs(ad - fd) / max(1.0, abs(ad), abs(fd))
                    ),
                )
            )
            for j, a in enumerate(low.itertuples()):
                rows.append(
                    (
                        name,
                        setting,
                        a.time_h,
                        j < 3,
                        a.sulfate_mol_per_L,
                        ans[j, 0],
                        a.log_c0_c,
                        ans[j, 1],
                    )
                )
            row = dict(
                network=name,
                setting=setting,
                loss=fit.loss,
                success=any(fit.converged),
                speed=p[-1],
                high_held_gf_rmse=np.sqrt(
                    np.mean((dg[~data.train] * data.scale[~data.train]) ** 2)
                ),
                high_held_ic_log_rmse=np.sqrt(
                    np.mean((di[~ic.train] * ic.scale[~ic.train]) ** 2)
                ),
                low_late_sulfate_rmse=np.sqrt(np.mean(errs[3:, 0] ** 2)),
                low_late_log_rmse=np.sqrt(np.mean(errs[3:, 1] ** 2)),
                **dict(zip(PARAMS, p[:-1])),
            )
            scores.append(row)
            print(setting, name, round(fit.loss, 2), p[-1], flush=True)
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    pd.DataFrame(
        rows,
        columns="network setting time_h early_block observed_sulfate_mol_L predicted_sulfate_mol_L observed_log_c0_c predicted_log_c0_c".split(),
    ).to_csv(out / "series.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
