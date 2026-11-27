"""Replay the prespecified whole-spectrum calibrations using individual merging droplets."""

from fit import HERE
from resolved_check import simulate
import json, tempfile
from pathlib import Path
import numpy as np


def replay(run, kind="joint", split="hold", mesh=576, n=65536, objective="native"):
    source = HERE / "final-fits" / f"{run}-18-1-0.05-{kind}-0.5-{split}-{objective}-0.05"
    info = json.loads((source / "fit.json").read_text())
    z = np.load(source / "prediction.npz")
    xt = z["t"]
    logs = z["logs"]
    temp = z["thermal"]
    flow = np.exp(z["flow"])
    from inverse import interpolator

    anchor = interpolator(np.log(np.arange(0.5, 351)), np.log([15.0, 35.0, 60.0]))

    def matrix(t, d):
        triple = np.zeros((3, 3))
        loc = np.triu_indices(3)
        triple[loc] = [np.interp(t, xt, s) for s in logs.T] if t >= 3 else 0.0
        triple[(loc[1], loc[0])] = triple[loc]
        return 10 ** (anchor @ triple @ anchor.T)

    def pairs(t):
        if kind == "smooth-pairs":
            return (1.0, 1.0)
        return 10 ** np.array([np.interp(t, xt, s) for s in logs.T]) if t >= 3 else (1.0, 1.0)

    def thermal(t):
        return float(np.interp(t, xt, temp))

    folder = HERE / "native" / f"{source.name}-{mesh}-{n}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fit.json").exists():
        return
    hidden = None
    if kind in ("reservoir", "entry"):
        nknots = len(info["knots"])
        params = info["x"][3 * nknots + 2 :]

        def rate(t):
            if kind == "reservoir" or t < 3:
                return 0.0
            return float(np.interp(t, info["knots"], params[1:]))

        hidden = (params[0], rate)
    with tempfile.TemporaryDirectory() as tmp:
        simulate(
            run,
            "jets",
            "standard",
            mesh=str(mesh),
            n=n,
            scales=flow,
            pair_schedule=pairs,
            thermal_schedule=thermal,
            output=tmp,
            matrix_schedule=matrix if kind == "smooth-pairs" else None,
            hidden_population=hidden,
        )
        raw = np.load(next(Path(tmp).glob("*.npz")))
        meta = json.loads(next(Path(tmp).glob("*.json")).read_text())
        full = raw["spectra"][:, :, 0, :80]
        np.savez_compressed(
            folder / "prediction.npz",
            t=raw["t"],
            d=raw["d"][:80],
            spec=full.mean(1, dtype=np.float64),
            ensembles=full,
            events=raw["events"],
        )
        meta.update(
            source=source.name,
            kind=kind,
            split=split,
            cadence=info["cadence"],
            width=1,
            knot=0.5,
            objective=objective,
            mesh=mesh,
        )
        (folder / "fit.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("kind")
    p.add_argument("--mesh", type=int, default=576)
    p.add_argument("--n", type=int, default=65536)
    p.add_argument("--split", default="hold")
    p.add_argument("--objective", default="native")
    replay(**vars(p.parse_args()))
