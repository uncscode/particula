"""Automatic differentiation through the chamber evolution with blocked observation masks."""

import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from observations import load, D
from inverse import HERE, Forward, torch, project, interpolator, train_mask


def fit(
    run,
    kind="joint",
    knot=0.5,
    split="hold",
    obj="bins",
    cadence=0.05,
    mesh=18,
    width=2,
    dt=0.05,
    iters=42,
    seed=2.5,
    tag="",
    source=None,
    shared_endpoints=False,
):
    start = time.monotonic()
    name = f"{run}-{mesh}-{width}-{dt}-{kind}-{knot}-{split}-{obj}-{cadence}" + tag
    folder = HERE / "final-fits" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fit.json").exists():
        return
    model = Forward(run, mesh, width, dt, "cuda:0")
    put = model.put
    t, y = load(run)
    keep = (t >= 3) & (t < 7)
    t = t[keep]
    y = y[keep]
    pos = interpolator(t, model.t)
    edges = (
        np.array([11, 20, 30, 40, 50, 65, 79])
        if obj == "bands"
        else np.r_[11, np.arange(12, 79, 2), 79]
    )
    if obj in ("native", "tail4", "tail16"):
        edges = np.arange(11, 80)
    # Average the published optical columns within each interval at their actual recorded times.
    block = np.floor((t - 3) / cadence + 1e-8).astype(int) if cadence else np.arange(len(t))
    groups = np.unique(block)
    use = np.stack([(block == g) / (block == g).sum() for g in groups])
    tt = use @ t
    truth = use @ y @ project(D, 1, edges)
    mask = train_mask(tt, split)
    observe = put((use @ pos)[mask])
    measured = put(truth[mask])
    select = put(project(model.model.d, width, edges))
    span = put(np.diff(edges).astype(float))
    floor = 0.005
    # Last usable knot precedes the forecast. The gap interpolates between training controls.
    if knot == -1:
        kt = t[train_mask(t, split)]
    elif knot == 0:
        kt = np.array([3.0])
    else:
        last = {"all": 7.0, "hold": 6.25, "early": 5.0}[split]
        kt = np.arange(3 + knot / 2, last, knot)
        if split == "hold":
            kt = kt[(kt < 5) | (kt >= 5.5)]
    if shared_endpoints:
        usable = t[train_mask(t, split)]
        kt = np.unique(np.r_[usable[0], kt, usable[-1]])
    basis = put(interpolator(model.t, kt) * (model.t >= 3)[:, None])
    n = len(kt)
    birth_kind = kind in ["reservoir", "entry"]
    variable = kind == "transport-time"
    pair_count = 6 if kind == "smooth-pairs" else 2
    joint = kind in ["joint", "transport-time", "smooth-pairs", "reservoir", "entry"]
    thermal_only = kind == "environment"

    def births(x):
        if kind == "entry":
            return x[-n - 1], basis @ x[-n:]
        return (x[-1], put(np.zeros(len(model.t)))) if birth_kind else None

    def core(x):
        return x[: -(1 + n if kind == "entry" else 1)] if birth_kind else x

    def unpack(x):
        x = core(x)
        if kind in ["standard", "environment"]:
            logs = put(np.zeros((len(model.t), 2)))
        else:
            logs = basis @ x[: pair_count * n].reshape(pair_count, n).T
        if joint or thermal_only:
            st = 0 if thermal_only else pair_count * n
            temp = basis @ x[st : st + n]
            flow = basis @ x[st + n :].reshape(2, n).T if variable else x[st + n : st + n + 2]
        else:
            temp = put(np.zeros(len(model.t)))
            flow = put(np.zeros(2))
        return logs, flow, temp

    def loss(x):
        pred = observe @ model.predict(*unpack(x), birth=births(x)) @ select
        # Smoothed generalized KL on concentration, with a common per-µm floor. No raw count likelihood is assumed.
        a = measured + floor * span
        b = pred + floor * span
        diverg = a * torch.log(a / b) - a + b
        if obj in ("tail4", "tail16"):
            extra = 3 if obj == "tail4" else 15
            weights = 1 + extra * put(((edges[:-1] >= 50) & (edges[1:] <= 65)).astype(float))
            diverg = diverg * weights
        value = diverg.sum(1).mean()
        if obj == "shape":
            pred_sum = pred.sum(1)
            obs_sum = measured.sum(1)
            cp = (pred / pred_sum[:, None].clamp(min=1e-12)).cumsum(1)
            co = (measured / obs_sum[:, None].clamp(min=1e-12)).cumsum(1)
            value = (
                value
                + 0.08 * ((cp - co).square() * span).sum(1).mean()
                + 0.2 * (torch.log((pred_sum + 0.1) / (obs_sum + 0.1)).square()).mean()
            )
        if birth_kind:
            reservoir, enter = births(x)
            value = value + 0.0002 * reservoir.square() + 0.0005 * enter.square().mean()
            if kind == "entry" and n > 1:
                value = (
                    value + 0.003 * ((x[-n:][1:] - x[-n:][:-1]).square() / put(np.diff(kt))).mean()
                )
            x = core(x)
        if kind != "standard":
            blocks = (
                x.reshape(-1, n)
                if variable
                else (x[:-2].reshape(-1, n) if joint or thermal_only else x.reshape(2, n))
            )
            if n > 1:
                value = value + 0.008 * (
                    (blocks[:, 1:] - blocks[:, :-1]).square() / put(np.diff(kt))[None, :]
                ).sum() / max(1, kt[-1] - kt[0])
            if variable:
                value = (
                    value
                    + 0.015 * blocks[2].square().mean()
                    + 0.015 * blocks[3:].square().sum(0).mean()
                )
            elif joint or thermal_only:
                value = value + 0.015 * (blocks[-1] ** 2).mean() + 0.015 * x[-2:].square().sum()
            if pair_count == 6:
                adjacent = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
                value = (
                    value
                    + 0.004
                    * torch.stack(
                        [(blocks[a] - blocks[b]).square().mean() for a, b in adjacent]
                    ).mean()
                )
            if not thermal_only:
                value = value + 0.0005 * (x[: pair_count * n] ** 2).mean()
        return value

    ic = [] if thermal_only else [seed] * (pair_count * n)
    bounds = [] if thermal_only else [(0.0, 5.0)] * (pair_count * n)
    if joint or thermal_only:
        ic += [0.0] * n + [0.0] * (2 * n if variable else 2)
        bounds += [(-1.0, 1.0)] * n + [(-3.0, 1.39)] * (2 * n if variable else 2)
    if birth_kind:
        ic += [1.0]
        bounds += [(0.0, 20.0)]
        if kind == "entry":
            ic += [1.0] * n
            bounds += [(0.0, 15.0)] * n

    def fun(v):
        x = put(v).requires_grad_()
        l = loss(x)
        l.backward()
        return float(l.detach()), x.grad.detach().cpu().numpy()

    if source:
        prev = json.loads(Path(source).read_text())
        old = np.array(prev["x"])
        if len(old) == len(ic):
            ic = old
        elif kind == prev["kind"] == "joint" and split == prev["split"]:
            old_kt = np.array(prev["knots"])
            ic = np.r_[
                np.concatenate([np.interp(kt, old_kt, b) for b in old[:-2].reshape(3, -1)]),
                old[-2:],
            ]
    if kind == "standard":
        x = np.zeros(2 * n)
        resinfo = {}
    else:
        res = minimize(
            fun,
            np.array(ic),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": iters, "maxls": 12, "ftol": 1e-8, "gtol": 1e-5},
        )
        x = res.x
        resinfo = dict(
            obj_value=float(res.fun),
            iterations=res.nit,
            calls=res.nfev,
            termination=str(res.message),
        )
    with torch.no_grad():
        pars = unpack(put(x))
        spec, bud = model.predict(*pars, diagnostics=True, birth=births(put(x)))
        asnp = lambda a: a.cpu().numpy()
        spec = asnp(spec)
        bud = asnp(bud)
        assert (
            bud[:, :2].max() < 1e-10
            and bud[:, 2].max() < 1e-8
            and bud[:, 3].max() < 1e-8
            and bud[:, 4].min() > 0
            and bud[:, 5].min() > -1e-12
        )
        np.savez_compressed(
            folder / "prediction.npz",
            t=model.t,
            d=model.model.d,
            spec=spec,
            logs=asnp(pars[0]),
            flow=asnp(pars[1]),
            thermal=asnp(pars[2]),
            ledger=bud,
        )
    info = dict(
        run=run,
        kind=kind,
        knot=knot,
        split=split,
        objective=obj,
        cadence=cadence,
        mesh=mesh,
        width=width,
        dt=dt,
        seed=seed,
        knots=kt.tolist(),
        x=np.asarray(x).tolist(),
        **resinfo,
        device=torch.cuda.get_device_name(),
        seconds=time.monotonic() - start,
        train_samples=int(mask.sum()),
    )
    (folder / "fit.json").write_text(json.dumps(info, indent=2))
    print(name, resinfo, round(info["seconds"], 2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--kind", default="joint")
    p.add_argument("--knot", type=float, default=0.5)
    p.add_argument("--split", default="hold")
    p.add_argument("--obj", default="bins")
    p.add_argument("--cadence", type=float, default=0.05)
    p.add_argument("--mesh", type=int, default=18)
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--iters", type=int, default=42)
    p.add_argument("--seed", type=float, default=2.5)
    p.add_argument("--tag", default="")
    p.add_argument("--source")
    fit(**vars(p.parse_args()))
