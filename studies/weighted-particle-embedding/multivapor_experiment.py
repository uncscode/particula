"""Fit transferable particle heads and score seen and unseen vapor counts."""

import argparse
from pathlib import Path

import numpy as np
import torch
from compact_inference import save
from dual_experiment import continuous, load_files, reporting
from moment_quadrature import decode, encode
from multivapor_transfer import (
    ExtrapolatedExchange,
    MultiExchange,
    RosenbrockExchange,
    SpeciesHead,
    from_checkpoint,
    transfer_class,
)
from node_transfer import write_table
from projection_study import digest
from quadrature import restore_arrays
from vapor_properties import materials


def encode_archive(file, d, k, spec=None, per_box=False):
    """Store size, dry composition and each volatile loading."""
    m, n = d["m"], d["n"]
    s = m.shape[-1] - 2
    if k:
        z, total = encode(m[:, :, [0, -2, -1]], n, k)
        base = 2 * k - 1
        profiles = []
        for lane in range(1, s):
            a, _ = encode(m[:, :, [lane, -2, -1]], n, k)
            profiles.append(a[:, base:].reshape(len(m), k, 5)[..., 3:])
        extra = (
            np.stack(profiles, 1) if profiles else np.empty((len(m), 0, k, 2))
        )
        payload = dict(
            shape=z.astype(np.float32),
            totals=total,
            extra=extra.astype(np.float32),
            inventory=(m[..., :s] * n[..., None]).sum(1),
        )
    else:
        payload = dict(m=m, n=n)
    chemistry = {}
    if spec is not None:
        if spec.species != s:
            raise ValueError("Material count differs from the initial state")
        chemistry = {
            key: getattr(spec, key)
            for key in (
                "molar",
                "density",
                "pressure",
                "diffusivity",
                "accommodation",
            )
        }
    save(
        file,
        dict(
            **payload,
            ids=d["ids"],
            levels=d["levels"] if per_box else d["levels"][::2],
            mask=d["reset_mask"] if per_box else d["reset_mask"][::2],
            time=d["time"],
            volume=d["volume"],
            nodes=k,
            species=(materials(s) if spec is None else spec).names,
            **chemistry,
            **({"per_box_levels": True} if per_box else {}),
        ),
    )


def reopen(file):
    """Decode each volatile on the same conservative dry particle basis."""
    with np.load(file) as a:
        k = int(a["nodes"])
        if k:
            cores = []
            n, dry = None, None
            for i in range(a["inventory"].shape[1]):
                shape = a["shape"]
                total = a["totals"]
                total[:, 1] = a["inventory"][:, i]
                if i:
                    shape[:, 2 * k - 1 :].reshape(len(total), k, 5)[..., 3:] = (
                        a["extra"][:, i - 1]
                    )
                z, _ = decode(shape, total, k)
                counts, mass = restore_arrays(z, total)
                if cores:
                    np.testing.assert_array_equal(counts, n)
                    np.testing.assert_array_equal(mass[..., 1:], dry)
                n, dry = counts, mass[..., 1:]
                cores.append(mass[..., 0])
            m = np.concatenate((np.stack(cores, -1), dry), -1)
        else:
            m, n = a["m"], a["n"]
        repeat = 1 if "per_box_levels" in a and a["per_box_levels"] else 2
        return (
            m,
            n / float(a["volume"]),
            a["levels"].repeat(repeat, 0),
            a["mask"].repeat(repeat, 0),
            a["time"],
        )


def references(root, split, s, families, bdf):
    """Align refinements and resolve each volatile signal independently."""
    folder = root / split
    hi, lo = [
        load_files(folder / f"v{s}-s{steps}-k12", families)
        for steps in (128, 64)
    ]
    for k in ("m", "n", "ids", "time", "levels", "reset_mask"):
        np.testing.assert_array_equal(hi[k], lo[k])
    diff = abs(hi["bulk"] - lo["bulk"])
    refs = [("native", hi["bulk"], diff[::2] + diff[1::2])]
    if bdf:
        source = load_files(folder / f"v{s}-s32-k12", families)
        fine, coarse = [
            continuous(
                root / "bdf" / f"v{s}", source["sources"], tol, "coupled"
            )[: len(hi["m"])]
            for tol in (1e-9, 1e-7)
        ]
        diff = abs(fine - coarse)
        refs.append(("BDF", fine, diff[::2] + diff[1::2]))
    hi["names"] = materials(s).names
    return hi, refs


def solvers(spec, paths, extrapolate=False, rosenbrock=False, higher=False):
    """Load the same learned head at every species count."""
    yield "physical", MultiExchange(spec), 0, 0, "", 0
    yield "rk3i3", MultiExchange(spec, order=3, iterations=3), 0, 0, "", 0
    if extrapolate:
        yield "extrapolated", ExtrapolatedExchange(spec), 0, 0, "", 0
    if rosenbrock:
        yield "ros2", RosenbrockExchange(spec), 0, 0, "", 0
    if higher:
        for n in (3, 4):
            yield f"ros{n}", transfer_class(ros_order=n)(spec), 0, 0, "", 0
    for path in paths:
        d = torch.load(path, weights_only=True, map_location="cpu")
        model = from_checkpoint(spec, d)
        yield (
            path.stem,
            model,
            path.stat().st_size,
            d["seed"],
            digest(path),
            d["parameters"],
        )


@torch.inference_mode()
def assess(
    root, out, split, counts, families, nodes, substeps, checkpoints, bdf
):
    """Measure free trajectories of every volatile at a given material count."""
    rows = []
    for s in counts:
        d, refs = references(root, split, s, families, bdf)
        folder = root / "archives" / split
        folder.mkdir(parents=True, exist_ok=True)
        for k in nodes:
            file = folder / f"v{s}-f{families}-n{k}.npz"
            encode_archive(file, d, k)
            m, n, levels, mask, t = reopen(file)
            np.testing.assert_allclose(
                (m * n[..., None]).sum(1),
                (d["m"] * d["n"][..., None]).sum(1) / d["volume"],
                rtol=2e-13,
            )
            m, n, levels, mask = map(torch.from_numpy, (m, n, levels, mask))
            for name, net, size, seed, sha, parameters in solvers(
                materials(s), checkpoints
            ):
                for steps in substeps:
                    w, g = net.rollout(m, n, levels, t, steps, mask)
                    wet = (w * n[:, None, :, None]).sum(2).numpy()
                    for ref, target, bound in refs:
                        old = len(rows)
                        reporting(
                            name,
                            size,
                            wet,
                            g.numpy(),
                            d,
                            target,
                            bound,
                            ref,
                            file,
                            f"{s}-vapors",
                            steps,
                            seed,
                            rows,
                        )
                        for r in rows[old:]:
                            r.update(
                                nodes=k,
                                checkpoint_sha256=sha,
                                parameters=parameters,
                                volatile_species=s,
                            )
            write_table(rows, out)
            print(s, k, len(rows), flush=True)


def prep(root, s, families, directory=None):
    """Use native training trajectories and training-only normalization."""
    folder = directory or root / "train" / f"v{s}-s64-k12"
    d = load_files(folder, families, True)
    for k in ("m", "n", "gas", "levels", "reset_mask", "volatile", "bulk"):
        d[k] = torch.from_numpy(d[k])
    d["n"] = d["n"] / d["volume"]
    d["leg_after"] = np.cumsum(np.diff(d["time"]) == 0)
    d["scale"] = d["bulk"].mean((0, 1)).clamp_min(1e-12)
    return d


def objective(net, d, rng, batch, window, substeps=1):
    """Warm the shared recurrent state and predict both members causally."""
    if substeps < 1:
        raise ValueError("Training needs a positive substep count")
    selected = rng.integers(len(d["ids"]), size=batch)
    ix = (2 * selected[:, None] + np.arange(2)).ravel()
    start = (
        0 if rng.random() < 0.3 else int(rng.integers(len(d["time"]) - window))
    )
    warm = max(start - 4, 0)
    m, n = d["m"][ix], d["n"][ix]
    ctx = net.static(m, n)
    q = d["volatile"][ix, warm] / ctx[0][..., None]
    gas = d["gas"][ix, warm]
    h = net.start(n)
    paths = []
    for i in range(warm, start + window):
        with torch.set_grad_enabled(i >= start):
            dt = d["time"][i + 1] - d["time"][i]
            if dt == 0:
                j = d["leg_after"][i]
                gas = torch.where(
                    d["reset_mask"][ix, j], d["levels"][ix, j], gas
                )
            else:
                step = gas.new_full((len(ix),), dt / substeps)
                for _ in range(substeps):
                    q, gas, h = net.step(q, gas, ctx, step, h)
        if i >= start:
            paths.append(q)
    pred = torch.stack(paths, 1) * ctx[0][:, None, :, None]
    ref = d["volatile"][ix, start + 1 : start + window + 1]
    node = (pred - ref) * n[:, None, :, None] / d["scale"]
    err = node.sum(2)
    return (
        err.abs().mean()
        + 2 * (err[1::2] - err[::2]).abs().mean()
        + node.abs().sum(2).mean()
    )


def train(
    root,
    out,
    counts,
    families,
    kinds,
    updates,
    seed,
    batch,
    window,
    chemistries=None,
    extrapolate=False,
    rosenbrock=False,
    ros_order=0,
):
    """Share one parameter set over the selected native vapor populations."""
    if chemistries:
        from material_bank import mixture

        specs = [
            mixture(s, c) for s, c in zip(counts, chemistries, strict=True)
        ]
        sets = [
            prep(root, s, families, root / "train" / f"c{c}-v{s}-s64-k8")
            for s, c in zip(counts, chemistries, strict=True)
        ]
    else:
        specs = [materials(s) for s in counts]
        sets = [prep(root, s, families) for s in counts]
    out.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        width = min(
            range(4, 129),
            key=lambda w: abs(
                sum(p.numel() for p in SpeciesHead(kind, w).parameters()) - 1600
            ),
        )
        config = dict(kind=kind, width=width)
        head = SpeciesHead(kind, width)
        cls = transfer_class(
            extrapolate=extrapolate, rosenbrock=rosenbrock, ros_order=ros_order
        )
        nets = [cls(spec, **config) for spec in specs]
        for net in nets:
            net.head = head
        opt = torch.optim.AdamW(head.parameters(), lr=0.0015)
        losses = []
        for i in range(updates):
            j = i % len(sets)
            loss = objective(nets[j], sets[j], rng, batch, window)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            losses.append(float(loss.detach()))
        label = "".join(str(s) for s in counts)
        if chemistries:
            label += "-chem" + "-".join(map(str, chemistries))
        if extrapolate:
            label += "-extrapolated"
        if rosenbrock:
            label += "-ros2"
        if ros_order:
            label += f"-ros{ros_order}"
        file = out / f"v{label}-{kind}-u{updates}-s{seed}.pt"
        tmp = file.with_suffix(".tmp")
        torch.save(
            dict(
                head=head.state_dict(),
                config=config,
                counts=counts,
                chemistries=chemistries,
                extrapolate=extrapolate,
                rosenbrock=rosenbrock,
                ros_order=ros_order,
                seed=seed,
                parameters=sum(p.numel() for p in head.parameters()),
                families=families,
                updates=updates,
                window=window,
                batch=batch,
                loss=float(np.mean(losses[-20:])),
            ),
            tmp,
        )
        tmp.replace(file)
        print(file.name, losses[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "assess"):
        p = subs.add_parser(action)
        p.add_argument("root", type=Path)
        p.add_argument("out", type=Path)
        p.add_argument(
            "--counts",
            nargs="+",
            type=int,
            default=[2, 3] if action == "train" else [1, 2, 3, 4],
        )
        p.add_argument(
            "--families", type=int, default=1536 if action == "train" else 80
        )
        if action == "train":
            method = p.add_mutually_exclusive_group()
            method.add_argument("--extrapolate", action="store_true")
            method.add_argument("--rosenbrock", action="store_true")
            method.add_argument(
                "--ros-order", type=int, choices=(3, 4), default=0
            )
            p.add_argument("--chemistries", nargs="+", type=int)
            p.add_argument(
                "--kinds",
                nargs="+",
                choices=["ff", "gru", "attention", "mixture"],
                default=["ff", "gru", "attention", "mixture"],
            )
            for k, v in (
                ("updates", 1800),
                ("seed", 4001),
                ("batch", 8),
                ("window", 8),
            ):
                p.add_argument("--" + k, type=int, default=v)
        else:
            p.add_argument("--split", default="validation")
            p.add_argument("--nodes", type=int, nargs="+", default=[0, 5])
            p.add_argument("--substeps", type=int, nargs="+", default=[1, 2, 4])
            p.add_argument("--checkpoints", type=Path, nargs="*", default=[])
            p.add_argument("--bdf", action="store_true")
    args = vars(parser.parse_args())
    torch.set_num_threads(1)
    {"train": train, "assess": assess}[args.pop("action")](**args)
