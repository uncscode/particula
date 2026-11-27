"""Fit shared vapor-transfer architectures and qualify both volatile
inventories.
"""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from dual_materials import DENSITY, MOLAR, NAMES
from dual_stages import build
from dual_transfer import DualExchange
from moment_quadrature import decode, encode
from node_fit import metrics_for_series
from node_transfer import write_table
from projection_study import digest
from quadrature import restore_arrays

WIDTHS = {
    "ff": 102,
    "gru": 14,
    "mixture": 12,
    "set": 28,
    "attention": 21,
    "spatiotemporal": 9,
}


def load(
    root, split, phase, steps, families, training=False, bins=16, samples=16
):
    """Read complete, source-identified native family histories."""
    folder = root / split / phase / f"s{steps}-b{bins}-t{samples}-cuda"
    return load_files(folder, families, training)


def completed_shards(folder):
    """Return completed native family archives in identifier order."""
    return sorted(
        p
        for p in folder.glob("families-*.npz")
        if not p.name.endswith(".tmp.npz")
    )


def load_files(folder, families, training=False):
    """Collect complete paired cases from a single resolution and schedule."""
    chunks = []
    for path in completed_shards(folder):
        with np.load(path) as a:
            values = dict(
                ids=a["ids"],
                m=a["mass0"],
                n=a["counts"],
                time=a["time"],
                levels=a["levels"].repeat(2, 0),
                bulk=a["bulk"].transpose(1, 0, 2),
                gas=a["gas"].transpose(1, 0, 2),
                volume=float(a["volume"]),
                sha=digest(path),
                stem=path.stem,
            )
            if "reset_mask" in a:
                values["reset_mask"] = a["reset_mask"].repeat(2, 0)
            else:
                values["reset_mask"] = np.ones_like(
                    values["levels"], dtype=bool
                )
            if training:
                values["volatile"] = a["volatile"].transpose(1, 0, 2, 3)
            chunks.append(values)
            if sum(len(c["ids"]) for c in chunks) >= families:
                break
    if not chunks:
        raise ValueError(f"Missing native dual-vapor histories in {folder}")
    out = {}
    for k in ("ids", "m", "n", "levels", "reset_mask", "bulk", "gas") + (
        ("volatile",) if training else ()
    ):
        out[k] = np.concatenate([c[k] for c in chunks])[
            : families * (1 if k == "ids" else 2)
        ]
    if len(np.unique(out["ids"])) != families:
        raise ValueError("Incomplete or repeated families")
    out["time"], out["volume"] = chunks[0]["time"], chunks[0]["volume"]
    for c in chunks:
        np.testing.assert_array_equal(c["time"], out["time"])
        if c["volume"] != out["volume"]:
            raise ValueError("Different box volumes")
    out["sources"] = [(c["stem"], c["sha"]) for c in chunks]
    return out


def continuous(folder, sources, tol, label):
    """Load independent continuous solutions with matching sources."""
    path = []
    for name, sha in sources:
        with np.load(folder / f"{name}-r{tol:.0e}-{label}.npz") as a:
            if str(a["source_sha256"]) != sha or float(a["tolerance"]) != tol:
                raise ValueError("Different BDF inputs")
            path.append(a["water"])
    return np.concatenate(path)


def save_population(
    file,
    d,
    nodes,
    per_box_levels=False,
    organic_profile=False,
    moment_dtype=np.float32,
):
    """Serialize the supplied state, forcing and material order."""
    m, n = d["m"], d["n"]
    if nodes:
        geometry, totals = encode(m[:, :, [0, 2, 3]], n, nodes)
        payload = dict(
            shape=geometry.astype(moment_dtype),
            totals=totals,
            organic=(m[..., 1] * n).sum(1),
        )
        if organic_profile:
            profile, _ = encode(m[:, :, [1, 2, 3]], n, nodes)
            payload["organic_profile"] = (
                profile[:, 2 * nodes - 1 :]
                .reshape(len(n), nodes, 5)[..., 3:]
                .astype(moment_dtype)
            )
    else:
        payload = dict(mass=m, counts=n)
    if "reset_mask" in d:
        payload["reset_mask"] = (
            d["reset_mask"] if per_box_levels else d["reset_mask"][::2]
        )
    with NamedTemporaryFile(
        dir=file.parent, suffix=".npz", delete=False
    ) as tmp:
        np.savez_compressed(
            tmp,
            **payload,
            ids=d["ids"],
            nodes=nodes,
            levels=d["levels"] if per_box_levels else d["levels"][::2],
            time=d["time"],
            volume=d["volume"],
            molar=MOLAR,
            density=DENSITY,
            species=NAMES,
        )
    Path(tmp.name).replace(file)


def reopen(file):
    """Decode the stored particle representation and both gas schedules."""
    with np.load(file) as a:
        k = int(a["nodes"])
        if k:
            shape, _ = decode(a["shape"], a["totals"], k)
            n, core = restore_arrays(shape, a["totals"])
            dry = core[..., 1:].sum(2)
            if "organic_profile" in a:
                code = a["shape"]
                code[:, 2 * k - 1 :].reshape(len(n), k, 5)[..., 3:] = a[
                    "organic_profile"
                ]
                totals = a["totals"]
                totals[:, 1] = a["organic"]
                org_shape, _ = decode(code, totals, k)
                org_n, org = restore_arrays(org_shape, totals)
                np.testing.assert_array_equal(org_n, n)
                np.testing.assert_array_equal(org[..., 1:], core[..., 1:])
                organic = org[..., 0]
            else:
                total_dry = (dry * n).sum(1)
                fraction = np.divide(
                    a["organic"],
                    total_dry,
                    out=np.zeros_like(total_dry),
                    where=total_dry > 0,
                )
                organic = dry * fraction[:, None]
            mass = np.concatenate(
                (core[..., :1], organic[..., None], core[..., 1:]), 2
            )
        else:
            mass, n = a["mass"], a["counts"]
        levels = a["levels"]
        if len(levels) != len(n):
            levels = levels.repeat(2, 0)
        return mass, n, levels, a["time"], float(a["volume"])


def reporting(
    model,
    shared,
    pred,
    gas,
    data,
    exact,
    bound,
    mask,
    file,
    phase,
    steps,
    seed,
    rows,
):
    """Score each vapor on resolved positive-lead differences."""
    times = np.broadcast_to(
        data["time"], (len(data["ids"]), len(data["time"]))
    ).copy()
    reset = times[0, 1:] == times[0, :-1]
    total = pred + gas
    np.testing.assert_allclose(np.diff(total, axis=1)[:, ~reset], 0, atol=2e-14)
    prescribed = data["levels"][:, 1:]
    if "reset_mask" in data:
        prescribed = np.where(
            data["reset_mask"][:, 1:], prescribed, gas[:, :-1][:, reset]
        )
    np.testing.assert_allclose(
        gas[:, 1:][:, reset], prescribed, rtol=0, atol=1e-16
    )
    np.testing.assert_allclose(np.diff(pred, axis=1)[:, reset], 0, atol=1e-16)

    if (pred < 0).any() or (gas < 0).any():
        raise ArithmeticError("Negative reservoir")
    for lane, name in enumerate(data.get("names", NAMES)):
        ref = exact[:, :, lane]
        signal = ref[1::2] - ref[::2]
        uncertainty = bound[:, :, lane]
        truth = dict(
            ref=(signal, uncertainty),
            water=torch.from_numpy(ref),
            time=torch.from_numpy(times.repeat(2, 0)),
        )
        stats = metrics_for_series(
            torch.from_numpy(pred[:, :, lane]),
            torch.from_numpy(gas[:, :, lane]),
            truth,
            True,
        )["all"]
        if data.get("bootstrap", False):
            from family_intervals import intervals

            stats.update(
                intervals(
                    pred[:, :, lane].reshape(len(times), 2, -1),
                    ref.reshape(len(times), 2, -1),
                    uncertainty,
                    times[0],
                    data["ids"],
                )
            )
        positive = np.diff(times, axis=1, prepend=times[:, :1]) > 0
        resolved = positive & (np.abs(signal) > 4 * uncertainty)
        error = pred[1::2, :, lane] - pred[::2, :, lane] - signal

        def mean_abs(a, mask):
            return float(np.abs(a[mask]).mean() * 1e6) if mask.any() else ""

        rows.append(
            dict(
                model=model,
                seed=seed,
                phase=phase,
                material=name,
                reference=mask,
                substeps=steps,
                record_bytes=file.stat().st_size,
                record_sha256=digest(file),
                shared_bytes=shared,
                mean_resolved_signal_mg_m3=mean_abs(signal, resolved),
                mean_resolved_contrast_error_mg_m3=mean_abs(error, resolved),
                wet_end_contrast_mae_mg_m3=float(
                    abs(error[:, np.flatnonzero(reset)[0]]).mean() * 1e6
                ),
                dry_end_contrast_mae_mg_m3=float(
                    abs(error[:, -1]).mean() * 1e6
                ),
                **stats,
            )
        )


def reference_panels(root, split, phase, families, continuous_reference):
    """Resolve material-wise uncertainty and the isolated-vapor control."""
    d = load(root, split, phase, 128, families)
    coarse = load(root, split, phase, 64, families)
    for k in ("ids", "m", "n", "levels", "time"):
        np.testing.assert_array_equal(d[k], coarse[k])
    diff = np.abs(d["bulk"] - coarse["bulk"])
    panels = [("native", d["bulk"], diff[::2] + diff[1::2])]
    controls = []
    if continuous_reference:
        source = load(root, split, phase, 32, families)
        source_files = source["sources"]
        folder = root / "bdf" / phase
        values = continuous(folder, source_files, 1e-9, "coupled")[
            : len(d["m"])
        ]
        loose = continuous(folder, source_files, 1e-7, "coupled")[: len(d["m"])]
        isolated = continuous(folder, source_files, 1e-9, "isolated")[
            : len(d["m"])
        ]
        dc = abs(values - loose)
        panels.append(("BDF", values, dc[::2] + dc[1::2]))
        for label, w in (
            ("full-BDF", values),
            ("independent-lanes-BDF", isolated),
            ("native128", d["bulk"]),
        ):
            cuts = np.r_[
                0,
                np.flatnonzero(np.diff(d["time"]) == 0) + 1,
                len(d["time"]),
            ]
            gas = np.empty_like(w)
            for j, (i, end) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
                gas[:, i:end] = (
                    d["levels"][:, j : j + 1] + w[:, i : i + 1] - w[:, i:end]
                )
            controls.append((label, w, gas))
    return d, panels, controls


@torch.inference_mode()
def assess(
    root,
    out,
    split,
    families,
    nodes,
    substeps,
    models,
    continuous_reference,
    bootstrap=False,
):
    """Compare joint and independent vapor forecasts."""
    rows = []
    dst = root / "archives" / split
    dst.mkdir(parents=True, exist_ok=True)
    methods = [("physical", DualExchange(), 0, 0)]
    for p in models:
        stored = torch.load(p, map_location="cpu", weights_only=True)
        model = build(stored["config"])
        model.load_state_dict(stored["model"])
        methods.append((p.stem, model, p.stat().st_size, stored["seed"]))
    for phase in ("together", "opposed"):
        d, panels, controls = reference_panels(
            root, split, phase, families, continuous_reference
        )
        d["bootstrap"] = bootstrap
        for n in nodes:
            file = dst / f"{phase}-n{n}.npz"
            save_population(file, d, n)
            m, counts, levels, t, vol = reopen(file)
            np.testing.assert_allclose(
                (m * counts[..., None]).sum(1),
                (d["m"] * d["n"][..., None]).sum(1),
                rtol=1e-13,
            )
            number = counts / vol
            for label, w, g in controls if n == 0 else []:
                for mask, target, bound in panels:
                    reporting(
                        label,
                        0,
                        w,
                        g,
                        d,
                        target,
                        bound,
                        mask,
                        file,
                        phase,
                        0,
                        0,
                        rows,
                    )
            for label, model, size, seed in methods:
                for s in substeps:
                    wet, g = model.rollout(
                        *map(torch.from_numpy, (m, number, levels)), t, s
                    )
                    pred = (
                        (wet * torch.from_numpy(number)[:, None, :, None])
                        .sum(2)
                        .numpy()
                    )
                    for mask, target, bound in panels:
                        reporting(
                            f"{label}-n{n}",
                            size,
                            pred,
                            g.numpy(),
                            d,
                            target,
                            bound,
                            mask,
                            file,
                            phase,
                            s,
                            seed,
                            rows,
                        )
            write_table(rows, out)


def training_panels(root, asynchronous, families):
    """Prepare independent paired native trajectories for fitting."""
    sets = []
    for phase in ("together", "opposed"):
        d = load(root, "train", phase, 64, families, True)
        sets.append(d)
    if asynchronous:
        sets.append(
            load_files(
                asynchronous / "train/conditioned/b1-s64-k16", families, True
            )
        )
    for d in sets:
        cuts = np.diff(d["time"]) == 0
        d["leg_after"] = np.cumsum(cuts)
        for k in ("m", "n", "bulk", "gas", "volatile", "levels", "reset_mask"):
            d[k] = torch.from_numpy(d[k])
        d["n"] = d["n"] / d["volume"]
    return sets


def train(
    root,
    out,
    families,
    updates,
    seed,
    kinds,
    window,
    batch,
    asynchronous=None,
    order=0,
    iterations=1,
):
    """Fit causal transfer corrections on paired native histories."""
    sets = training_panels(root, asynchronous, families)
    out.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        config = dict(kind=kind, width=WIDTHS[kind])
        if order:
            config.update(order=order, iterations=iterations)
        model = build(config)
        opt = torch.optim.AdamW(model.parameters(), lr=0.002)
        losses = []
        for i in range(updates):
            d = sets[i % len(sets)]
            idx = (
                rng.integers(families, size=batch)[:, None] * 2 + np.arange(2)
            ).ravel()
            t = d["time"]
            start = (
                0 if rng.random() < 0.2 else int(rng.integers(len(t) - window))
            )
            warm = max(0, start - 4)
            m, n = d["m"][idx], d["n"][idx]
            ctx = model.static(m, n)
            dry = ctx[0]
            q = d["volatile"][idx, warm] / dry[..., None]
            gas = d["gas"][idx, warm]
            memory = model.start(n)
            pred = []
            for j in range(warm, start + window):
                with torch.set_grad_enabled(j >= start):
                    dt = t[j + 1] - t[j]
                    if dt == 0:
                        leg = d["leg_after"][j]
                        gas = torch.where(
                            d["reset_mask"][idx, leg],
                            d["levels"][idx, leg],
                            gas,
                        )
                    else:
                        q, gas, memory = model.step(
                            q, gas, ctx, gas.new_full((len(gas),), dt), memory
                        )
                    if j >= start:
                        pred.append(q)
            paths = torch.stack(pred, 1)
            weights = ctx[3][:, None, :, None]
            wd = weights.sum(2).clamp_min(1e-20)
            scales = m.new_tensor([1.0, 0.12])
            sl = slice(start + 1, start + window + 1)
            delta = (paths * weights).sum(2) - d["bulk"][idx, sl]
            individual = delta / (wd * scales)
            paired = individual[1::2] - individual[::2]
            node = (
                paths - d["volatile"][idx, sl] / dry[:, None, :, None]
            ).abs()
            loss = (
                individual.abs().mean()
                + 2 * paired.abs().mean()
                + ((node * weights).sum(2) / (wd * scales)).mean()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            losses.append(float(loss.detach()))
        prefix = f"rk{order}i{iterations}-" if order else ""
        file = out / f"{prefix}{kind}-n{families}-u{updates}-s{seed}.pt"
        tmp = file.with_suffix(".tmp")
        torch.save(
            dict(
                model=model.state_dict(),
                config=config,
                seed=seed,
                updates=updates,
                train_families=families,
                asynchronous=bool(asynchronous),
                window=window,
                batch=batch,
                loss=float(np.mean(losses[-20:])),
            ),
            tmp,
        )
        tmp.replace(file)
        print(file.name, losses[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("assess", "train"):
        a = sub.add_parser(name)
        a.add_argument("root", type=Path)
        a.add_argument("out", type=Path)
        a.add_argument(
            "--families", type=int, default=96 if name == "assess" else 1024
        )
        if name == "assess":
            a.add_argument("--models", type=Path, nargs="*", default=[])
            a.add_argument(
                "--split",
                choices=("validation", "test", "challenge"),
                default="validation",
            )
            a.add_argument("--nodes", type=int, nargs="+", default=[0, 3, 5, 8])
            a.add_argument(
                "--substeps", type=int, nargs="+", default=[1, 2, 4, 8]
            )
            a.add_argument("--bootstrap", action="store_true")
            a.add_argument("--continuous-reference", action="store_true")
        else:
            a.add_argument("--order", type=int, choices=(0, 2, 3), default=0)
            a.add_argument("--iterations", type=int, default=1)
            a.add_argument("--asynchronous", type=Path)
            a.add_argument("--updates", type=int, default=1200)
            a.add_argument("--seed", type=int, default=3101)
            a.add_argument("--window", type=int, default=8)
            a.add_argument("--batch", type=int, default=12)
            a.add_argument(
                "--kinds",
                nargs="+",
                choices=WIDTHS,
                default=["ff", "gru", "mixture"],
            )
    args = vars(p.parse_args())
    torch.set_num_threads(1)
    {"assess": assess, "train": train}[args.pop("action")](**args)
