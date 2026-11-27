"""Forecast two-vapor responses from a present record and a measured history."""

import argparse
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from dual_experiment import continuous, load_files, reporting
from dual_stages import DualStages
from history_encoder import Recovery, observed
from history_population import prehistory, reconstruct
from node_transfer import write_table
from projection_study import digest


def predicted_history(sample, coeff):
    """Penalize individual candidate states with physically impossible pasts."""
    d = dict(sample, coef=coeff)
    try:
        return prehistory(d)[0]
    except AssertionError:
        # Refine the independent forward/reverse check for stiff candidates.
        try:
            return prehistory(d, subdivisions=4)[0]
        except (AssertionError, ArithmeticError):
            pass
    except ArithmeticError:
        pass
    if len(coeff) == 1:
        return np.full((1, 8, 2), np.nan)
    stop = len(coeff) // 2
    return np.concatenate(
        [
            predicted_history({k: v[ix] for k, v in sample.items()}, coeff[ix])
            for ix in (slice(0, stop), slice(stop, None))
        ]
    )


def physical_inverse(d, history, frames, iterations=8):
    """Fit two loading gradients to observations with a damped local solve."""
    # Fit from the dry record, bulk means and preceding gas.
    sample = {k: d[k] for k in ("fixed", "number", "basis", "mean", "previous")}

    def residual(a):
        sim = predicted_history(sample, a)[:, -frames:]
        change = (sim - sim[:, -1:]) * 1e6
        r = (change - history[:, -frames:]).reshape(len(a), -1)
        return np.nan_to_num(r, nan=1e8)

    best = np.zeros((len(history), 2))
    scores = np.full(len(best), np.inf)
    for start in (0.0, -0.45, 0.45):
        a = np.full_like(best, start)
        for _ in range(iterations):
            r = residual(a)
            dx = 1e-4
            jac = np.stack(
                [(residual(a + dx * v) - r) / dx for v in np.eye(2)], -1
            )
            gram = jac.transpose(0, 2, 1) @ jac
            grad = jac.transpose(0, 2, 1) @ r[..., None]
            ridge = np.maximum(gram.trace(axis1=1, axis2=2) * 1e-8, 1e-12)
            step = np.linalg.solve(
                gram + np.eye(2) * ridge[:, None, None], grad
            )[..., 0]
            trial = (a - np.clip(step, -0.4, 0.4)).clip(-0.90, 0.90)
            improved = (residual(trial) ** 2).sum(1) < (r**2).sum(1)
            a = np.where(improved[:, None], trial, (a + trial) * 0.5)
        err = (residual(a) ** 2).sum(1)
        best = np.where((err < scores)[:, None], a, best)
        scores = np.minimum(scores, err)
    return best


def native_targets(root, split, families, bdf):
    """Match all family IDs, inputs and times across timestep refinements."""
    src = root / "native" / split
    hi, lo = [load_files(src / f"s{s}", families) for s in (128, 64)]
    for k in ("m", "n", "ids", "levels", "time"):
        np.testing.assert_array_equal(hi[k], lo[k])
    err = abs(lo["bulk"] - hi["bulk"])
    panels = [("native", hi["bulk"], err[::2] + err[1::2])]
    if bdf:
        src32 = load_files(src / "s32", families)
        fine, coarse = [
            continuous(root / "bdf" / split, src32["sources"], tol, "coupled")
            for tol in (1e-9, 1e-7)
        ]
        err = abs(fine - coarse)
        panels.append(("BDF", fine, err[::2] + err[1::2]))
    return hi, panels


def methods(d, paths, noise, frames, inverse):
    """Compare present, measured-history, inverse and oracle controls."""
    static, x = observed(d)
    eps = np.random.default_rng(4912).normal(0, noise, x.shape)
    eps[:, -1] = 0
    x = x + torch.tensor(eps, dtype=x.dtype)
    yield "exact-allocation", d["coef"], 0, 0, "", 0
    yield "present-only", np.zeros_like(d["coef"]), 0, 0, "", 1
    if inverse:
        for k in frames:
            yield (
                f"inverse-h{k}",
                physical_inverse(d, x.numpy(), k),
                0,
                0,
                "",
                k,
            )
    for file in paths:
        s = torch.load(file, weights_only=True, map_location="cpu")
        model = Recovery(**s["config"])
        model.load_state_dict(s["model"])
        pred = model(static, x).numpy()
        yield (
            file.stem,
            pred,
            file.stat().st_size,
            s["seed"],
            digest(file),
            1 if model.kind == "instant" else model.frames,
        )
        if (
            model.kind in ("mixture", "gru")
            and model.frames == 8
            and s["seed"] == 4201
        ):
            # Exchange the two member histories within each matched family.
            ix = np.arange(len(x)).reshape(-1, 2)[:, ::-1].ravel().copy()
            yield (
                file.stem + "-swapped",
                model(static, x[ix]).numpy(),
                file.stat().st_size,
                s["seed"],
                digest(file),
                model.frames,
            )


def estimated(paths, ids, source_sha, sigma):
    """Accept only estimates for this input, noise and family order."""
    for path in paths:
        with np.load(path) as a:
            if float(a["noise_mg_m3"]) != sigma:
                continue
            if str(a["input_sha256"]) != source_sha:
                raise ValueError("A recovered state belongs to another source")
            np.testing.assert_array_equal(a["ids"], ids)
            coeff = a["coef"]
            if (
                coeff.shape != (2 * len(ids), 2)
                or not np.isfinite(coeff).all()
                or abs(coeff).max() > 1
            ):
                raise ValueError("Invalid recovered allocation")
            yield (
                path.stem,
                coeff,
                int(a.get("shared_bytes", 0)),
                int(a.get("seed", 0)),
                str(a.get("checkpoint_sha256", "")),
                int(a["history_frames"]),
            )


def archive(out, d, coeff, used, sha):
    """Reopen the forecast-time inputs and reconstructed initial allocation."""
    dry = d["fixed"].sum(-1)
    np.testing.assert_allclose(
        (d["number"] * dry * d["basis"]).sum(1), 0.0, atol=2e-18
    )
    np.savez_compressed(
        out,
        fixed=d["fixed"],
        number=d["number"],
        basis=d["basis"],
        mean=d["mean"],
        levels=d["levels"],
        coef=coeff,
        observations_used=used,
        model_sha256=sha,
    )
    with np.load(out) as a:
        mass = reconstruct(a, a["coef"])
        number = a["number"]
        levels = a["levels"].repeat(2, 0)
        target = a["mean"] * (number * dry).sum(1, keepdims=True)
    np.testing.assert_allclose(
        (mass[..., :2] * number[..., None]).sum(1), target, rtol=1e-13
    )
    return tuple(map(torch.from_numpy, (mass, number, levels)))


@torch.inference_mode()
def score(
    root, record, out, paths, split, noise, frames, inverse, bdf, estimates=()
):
    """Score the first resolved forecast contrast after a measured history."""
    with np.load(record) as f:
        d = dict(f)
    data, panels = native_targets(root, split, len(d["ids"]), bdf)
    np.testing.assert_array_equal(d["ids"], data["ids"])
    np.testing.assert_array_equal(reconstruct(d, d["coef"]), data["m"])
    folder = root / "forecast-records" / split / out.stem
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    model = DualStages(3, 3)
    for sigma in noise:
        for label, coef, size, seed, sha, used in chain(
            methods(d, paths, sigma, frames, inverse),
            estimated(estimates, d["ids"], digest(record), sigma),
        ):
            file = folder / f"{label}-noise{sigma:g}.npz"
            m, n, levels = archive(file, d, coef, used, sha)
            wet, g = model.rollout(m, n, levels, data["time"], 4)
            bulk = (wet * n[:, None, :, None]).sum(2).numpy()
            for ref, truth, bound in panels:
                before = len(rows)
                reporting(
                    label,
                    size,
                    bulk,
                    g.numpy(),
                    data,
                    truth,
                    bound,
                    ref,
                    file,
                    "wet-dry",
                    4,
                    seed,
                    rows,
                )
                for r in rows[before:]:
                    lane = 0 if r["material"] == "water" else 1
                    r.update(
                        history_frames=used,
                        noise_mg_m3=sigma,
                        checkpoint_sha256=sha,
                        allocation_mae=float(
                            np.abs(coef[:, lane] - d["coef"][:, lane]).mean()
                        ),
                        allocation_sign=float(
                            np.mean(coef[:, lane] * d["coef"][:, lane] > 0)
                        ),
                        past_span_us=(max(used - 1, 0)) * 5,
                    )
            write_table(rows, out)
        print(sigma, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for f in ("root", "record", "out"):
        p.add_argument(f, type=Path)
    p.add_argument("--paths", type=Path, nargs="*", default=[])
    p.add_argument("--estimates", type=Path, nargs="*", default=[])
    p.add_argument("--split", default="validation")
    p.add_argument(
        "--noise", type=float, nargs="+", default=[0, 0.001, 0.01, 0.1]
    )
    p.add_argument("--frames", type=int, nargs="+", default=[2, 8])
    p.add_argument("--inverse", action="store_true")
    p.add_argument("--bdf", action="store_true")
    torch.set_num_threads(1)
    score(**vars(p.parse_args()))
