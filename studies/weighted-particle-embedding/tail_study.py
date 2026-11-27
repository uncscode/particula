"""Measure vapor response to a change in composition tails."""

import argparse
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from forecast import read_encoder
from measures import observables
from moment_quadrature import decode as moment_decode
from moment_quadrature import encode as moment_encode
from multicycle_storage import forcing
from node_fit import metrics_for_series
from node_transfer import write_table
from projection_study import digest
from quadrature import restore_arrays
from reduced_exchange import rollout
from tail_quadrature import decode, encode


def read(directory):
    """Read raw initial states and reference trajectories in family order."""
    groups = []
    for f in sorted(directory.glob("families-*.npz")):
        with np.load(f) as a:
            groups.append(
                dict(
                    ids=a["ids"],
                    m=a["mass"][0],
                    n=a["counts"],
                    levels=a["gas"][
                        [0, np.flatnonzero(np.diff(a["time"]) == 0)[0] + 1]
                    ].T,
                    time=a["time"],
                    w=a["water"].T,
                    density=a["density"],
                    volume=float(a["volume"]),
                )
            )
    if not groups:
        raise ValueError(f"No native tail histories in {directory}")
    data = {
        k: np.concatenate([g[k] for g in groups])
        for k in ("ids", "m", "n", "levels", "w")
    }
    for k in ("time", "density", "volume"):
        data[k] = groups[0][k]
        for item in groups:
            np.testing.assert_array_equal(data[k], item[k])
    if len(np.unique(data["ids"])) != len(data["ids"]):
        raise ValueError("Repeated family")
    return data


def continuous_water(source, folder, tolerance, expected_ids):
    """Verify independently integrated boxes against the original inputs."""
    groups, ids = [], []
    for p in sorted(source.glob("families-*.npz")):
        with np.load(folder / f"{p.stem}-r{tolerance:.0e}.npz") as a:
            if digest(p) != str(a["source_sha256"]) or a["rtol"] != tolerance:
                raise ValueError("Continuous composition reference changed")
            groups.append(a["water"])
            ids.append(a["ids"])
    np.testing.assert_array_equal(np.concatenate(ids), expected_ids)
    return np.concatenate(groups)


def physical(mass, counts, gas, force, volume, substeps):
    """Forecast positive water and gas on the reduced population."""
    water, g = rollout(
        *[torch.as_tensor(v) for v in (mass, counts / volume, gas, force)],
        substeps,
        True,
        True,
    )
    return (water * torch.as_tensor(counts / volume)[:, None]).sum(-1), g


def score(
    name, archive, shared, prediction, gas, truth, times, record, identity
):
    """Report the paired signal, forecast error and actual serialized cost."""
    rows = metrics_for_series(prediction, gas, truth, True)
    guess = (prediction[1::2] - prediction[::2]).numpy()
    target, bound = truth["ref"]
    positive = np.diff(times, axis=1, prepend=times[:, :1]) > 0
    keep = positive & (abs(target) > 4 * bound)
    resolved = target[keep]
    return dict(
        model=name,
        archive_bytes=archive.stat().st_size,
        shared_bytes=shared,
        archive_sha256=digest(archive),
        model_sha256=identity,
        max_abs_shape_coordinate_difference=np.abs(
            record[1::2] - record[::2]
        ).max(),
        resolved_observations=int(keep.sum()),
        mean_resolved_signal_mg_m3=np.abs(resolved).mean() * 1e6
        if len(resolved)
        else "",
        mean_resolved_contrast_error_mg_m3=np.abs(guess[keep] - resolved).mean()
        * 1e6
        if len(resolved)
        else "",
        positive_lead_sign_fraction=(guess[keep] * resolved > 0).mean()
        if len(resolved)
        else "",
        **rows["all"],
    )


def store_population(d, b, c):
    """Encode a native, global or size-conditioned population."""
    if c:
        stored = dict(record=encode(d["m"], d["n"], b, c))
    elif b:
        geom, totals = moment_encode(d["m"], d["n"], b)
        stored = dict(record=geom.astype(np.float32), totals=totals)
    else:
        stored = dict(record=d["m"], counts=d["n"])
    return stored


def read_population(a, b, c):
    """Decode the saved population at its requested resolution."""
    raw = a["record"]
    if c:
        m, n = decode(raw)
    elif b:
        sh, _ = moment_decode(raw, a["totals"], b)
        n, m = restore_arrays(sh, a["totals"])
    else:
        n, m = a["counts"], raw
    return m, n


@torch.inference_mode()
def compare(
    root,
    directory,
    prefix,
    out,
    bins,
    classes,
    substeps,
    checkpoints,
    families,
    split="validation",
    continuous=None,
):
    """Score the stored representations against the finest native run."""
    folder = root / "composition-tails"
    refs = [read(folder / f"{prefix}{s}" / split) for s in (32, 64, 128)]
    d = refs[-1]
    if len(d["ids"]) != families:
        raise ValueError("Incomplete fine native ensemble")
    for a in refs[:-1]:
        for k in ("m", "n", "levels", "time", "ids"):
            np.testing.assert_array_equal(a[k], d[k])
    times = np.broadcast_to(d["time"], (families, len(d["time"]))).copy()
    native = d["w"]
    err32 = np.abs(refs[0]["w"] - refs[1]["w"])
    native_diff = np.abs(native - refs[-2]["w"])
    diff = native_diff
    if continuous is not None:
        source = folder / f"{prefix}32" / split
        d["w"] = continuous_water(source, continuous, 1e-10, d["ids"])
        loose = continuous_water(source, continuous, 1e-8, d["ids"])
        diff = np.abs(d["w"] - loose)
    pair = d["w"][1::2] - d["w"][::2]
    bound = diff[::2] + diff[1::2]
    truth = dict(
        time=torch.from_numpy(times.repeat(2, 0)),
        water=torch.from_numpy(d["w"]),
        ref=(pair, bound),
    )
    levels = d["levels"][::2]
    np.testing.assert_array_equal(levels, d["levels"][1::2])
    samples = (times.shape[1] - 2) // 2
    common = dict(
        ids=d["ids"],
        levels=levels,
        steps_per_leg=samples,
        dt=times[0, 1],
        volume=d["volume"],
        density=d["density"],
    )

    def start(a):
        np.testing.assert_array_equal(a["ids"], d["ids"])
        return (
            a["levels"][:, 0].repeat(2),
            forcing(
                np.full((families, 2), int(a["steps_per_leg"])),
                a["levels"],
                float(a["dt"]),
            ),
            float(a["volume"]),
        )

    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = [("native-slots", 0, 0), ("original-weighted-moments", 5, 0)]
    specs.extend((f"size{b}-class{c}", b, c) for b in bins for c in classes)
    for name, b, c in specs:
        path = directory / (name + ".npz")
        stored = store_population(d, b, c)
        np.savez_compressed(path, **common, **stored)
        with np.load(path) as a:
            raw = a["record"]
            gas, force, vol = start(a)
            m, n = read_population(a, b, c)
            np.testing.assert_allclose(
                (m * n[..., None]).sum(1),
                (d["m"] * d["n"][..., None]).sum(1),
                rtol=1e-12,
                atol=1e-28,
            )
            w, g = physical(m, n, gas, force, vol, substeps)
            rows.append(score(name, path, 0, w, g, truth, times, raw, ""))
    for cp in checkpoints:
        model, info = load_model(root / cp, "cpu")
        encoder_path = root / info["encoder"]
        if info.get("encoder_sha256", digest(encoder_path)) != digest(
            encoder_path
        ):
            raise ValueError("Frozen neural encoder changed")
        enc = read_encoder(encoder_path, "cpu")
        x, p, _ = observables(
            d["m"],
            d["n"],
            d["density"],
            d["volume"],
            d["levels"][:, 0],
            enc.grid,
        )
        z = enc.encode(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(d["n"], dtype=torch.float32),
        )
        path = directory / (Path(cp).stem + ".npz")
        np.savez_compressed(
            path,
            **common,
            record=z.numpy(),
            physical=p,
            encoder_sha256=digest(encoder_path),
            checkpoint_sha256=digest(root / cp),
        )
        with np.load(path) as a:
            _, force, vol = start(a)
            shape = torch.as_tensor(a["record"])
            _, future = model.rollout(
                (shape - model.z_mean) / model.z_std,
                torch.as_tensor(a["physical"]),
                torch.as_tensor(force),
            )
            rows.append(
                score(
                    Path(cp).stem,
                    path,
                    (root / cp).stat().st_size + encoder_path.stat().st_size,
                    future[:, :, 1],
                    future[:, :, 4],
                    truth,
                    times,
                    a["record"],
                    digest(root / cp),
                )
            )
    for r in rows:
        r.update(
            reference="BDF" if continuous is not None else "native",
            mean_member_reference_change_mg_m3=1e6 * diff[:, 1:].mean(),
            mean_member_native_gap_mg_m3=1e6
            * abs(d["w"] - native)[:, 1:].mean(),
            mean_member_last_halving_mg_m3=1e6 * native_diff[:, 1:].mean(),
            mean_member_prior_halving_mg_m3=1e6 * err32[:, 1:].mean(),
            physical_substeps=substeps,
        )
    write_table(rows, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("root", "directory", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--continuous", type=Path)
    p.add_argument(
        "--split",
        choices=("validation", "test", "challenge"),
        default="validation",
    )
    p.add_argument("--prefix", default="refinement-s")
    p.add_argument("--bins", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--substeps", type=int, default=12)
    p.add_argument("--families", type=int, default=64)
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=["first-full-cycle.pt", "temporal-recursive-pilot-w8-s1831.pt"],
    )
    torch.set_num_threads(1)
    compare(**vars(p.parse_args()))
