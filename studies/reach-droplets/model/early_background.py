"""Fit the six early diameter bands jointly in the two expansions on the CPU."""

import json
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import qmc
from geometry import BOUNDS, H, airflow, geom, inputs, optical, vapor_chord
from workspace import WORK

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
T = torch.as_tensor


def forward_setup(run, mode, step=0.05, width=2, offset=0, flow_case="standard"):
    nr, nz, radial, v, diff, circ = geom(mode)
    multiplier = airflow(run)
    if flow_case == "reverse":
        circ = (v[:, None] * circ.T) / v[None, :]
    if flow_case == "equal-flow":
        multiplier = 1.0
    d = np.arange(width / 2, 120, width)
    times, p, base, chi, anomaly, speed, loss, init, bands, targets = inputs(run, mode, d, step)
    optic = T(optical(mode, d, offset))
    fraction = T(v[:, None])
    base = T(base[:, None])
    chi = T(chi[:, None])
    anomaly = T(anomaly)
    diameter = T(d)
    bins = len(d)
    survive = T(np.exp(-step * loss))
    fall = T(-np.expm1(-step * speed * nz / H))
    height = int(round(0.25 / step))
    mask = (times < 5).sum()

    def forward(x, train=False, retain=False):
        temp = base + x[0] + (0 if mode == "one" else x[1]) * anomaly
        C = temp - 273.15
        es = 611.21 * torch.exp((18.678 - C / 234.5) * C / (257.14 + C))
        diffv = 2.11e-5 * (temp / 273.15) ** 1.94 * 101325 / p
        k = 0.024 * (temp / 273.15) ** 0.9
        res = 1000 * (
            461.5 * temp / (diffv * es) + (2.5e6 / (461.5 * temp) - 1) * 2.5e6 / (k * temp)
        )
        growth = 8e12 * (chi * p / es - 1) / res
        if mode == "one":
            trans = T([[1.0]])
            profile = torch.ones(1, 1)
        else:
            trans = torch.matrix_exp(
                step * multiplier * (torch.exp(x[2]) * T(diff) + torch.exp(x[3]) * T(circ))
            )
            profile = torch.exp(
                x[4] * T(np.tile((radial - radial[0]) / (radial[-1] - radial[0]), nz)[:, None])
            )
        pop = fraction * profile * T(init)[None, :] / (optic * profile).sum(0)
        values = []
        fields = []
        transported = None
        coupled = flow_case == "vapor-transport"
        chord = T(vapor_chord(mode))
        unit_mass = 1000 * np.pi / 6 * 1e-12
        vapor_to_mass = p * 0.01801528 / (8.314462618 * temp)
        # A single uniform correction constrains the measured chord after gas exchange.
        # Local anomalies persist. Liquid water transfer uses continuous droplet diameters.
        wet_rate = 8e12 * p / (res * es)

        for i, g in enumerate(growth[: mask if train else len(times)]):
            if coupled:
                factor = vapor_to_mass[i]
                gas = factor * chi[i] if transported is None else transported
                path_mean = (chord * gas).sum() / (chord * factor).sum()
                gas = gas + factor * (chi[i] - path_mean)
                g = g + wet_rate[i] * (gas / factor - chi[i])

            new2 = diameter**2 + step * g[:, None]
            if coupled:
                water = (
                    unit_mass
                    * ((new2.clamp(min=0) ** 1.5 - diameter**3) * pop).sum(1)
                    / fraction[:, 0]
                )
                transported = (trans @ ((gas - water) * fraction[:, 0])) / fraction[:, 0]
                if not train:
                    assert transported.min() > 0
            pos = torch.sqrt(new2.clamp(min=1.0e-10)) / width - 0.5
            lower = pos.floor().long()
            alpha = pos - lower

            def deposit(j, weight):
                source = pop * weight * ((j >= 0) & (j < bins) & (new2 > 0))
                return torch.zeros_like(pop).scatter_add(1, j.clamp(0, bins - 1), source)

            adv = (deposit(lower, 1 - alpha) + deposit(lower + 1, alpha)) * survive
            settled = adv * (1 - fall)
            if nz > 1:
                settled = torch.cat((settled[:-nr] + adv[nr:] * fall, settled[-nr:]), dim=0)
            end = trans @ settled
            mid = (pop + end) / 2 / fraction
            values.append((mid * optic).sum(0))
            if retain:
                fields.append(mid)
            pop = end
        spectra = torch.stack(values)
        byq = spectra.reshape(-1, height, bins).mean(1) @ T(bands.T)
        if retain:
            return byq.detach().numpy(), torch.stack(fields).detach().numpy()
        return byq

    # Six bands, absolute concentration; stabilize sparse bands. Scales derived solely from calibration.
    scale = np.maximum(0.20, targets[1:9].mean(0))

    def objective(arr):
        x = torch.tensor(arr, requires_grad=True)
        z = (forward(x, True)[1:] - T(targets[1:9])) / T(scale)
        score = (z * z).mean()
        score.backward()
        return float(score.detach()), x.grad.numpy()

    return forward, objective, (times, d, v, bands, targets, scale)


RUNS = ("forced", "unforced")


def calibrate():
    keys = [[0, 1, 6, 7, 2], [3, 4, 6, 7, 5]]
    bounds = np.array([BOUNDS[i] for i in [0, 1, 4, 0, 1, 4, 2, 3]])
    points = qmc.Sobol(8, scramble=True, seed=1185).random_base2(7)
    case = flow_case = "vapor-transport"
    records = []
    setup = [forward_setup(run, "local", flow_case=flow_case) for run in RUNS]

    def fun(x, grad):
        total = 0.0
        deriv = np.zeros(8)
        for i, (_, obj, info) in enumerate(setup):
            if grad:
                val, g = obj(x[keys[i]])
                deriv[keys[i]] += g / 2
            else:
                with torch.no_grad():
                    y = setup[i][0](T(x[keys[i]]), True).numpy()[1:]
                    val = np.mean(((y - info[-2][1:9]) / info[-1]) ** 2)
            total += val / 2
        return (total, deriv) if grad else total

    cloud = bounds[:, 0] + points * (bounds[:, 1] - bounds[:, 0])
    scores = np.array([fun(x, False) for x in cloud])
    top = np.argsort(scores)[:5]
    for rank, ix in enumerate(top):
        res = minimize(
            lambda x: fun(x, True),
            cloud[ix],
            bounds=bounds,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 90, "ftol": 1e-9, "maxls": 16},
        )
        records.append(
            dict(
                case=case,
                rank=rank,
                initial=scores[ix],
                objective=res.fun,
                iterations=res.nit,
                calls=res.nfev,
                x=res.x.tolist(),
                message=res.message,
            )
        )
    best = min([r for r in records if r["case"] == case], key=lambda r: r["objective"])
    result = {run: np.array(best["x"])[keys[i]].tolist() for i, run in enumerate(RUNS)}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "early-background.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    calibrate()
