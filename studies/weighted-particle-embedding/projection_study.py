"""Evaluate learned moment projections after reloading each archive."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from conditional_geometry import (
    fit_geometry,
    geometry,
    read_geometry,
    store_geometry,
)
from mixture_profile import (
    decode_mixture,
    encode_mixture,
    fit_mixture,
    pack_routes,
    unpack_routes,
)
from moment_quadrature import decode, encode
from multicycle_storage import forcing, inputs
from node_fit import metrics_for_series
from projected_moments import fit, project, recover, training_records
from quadrature import restore_arrays
from quadrature_validation import load_reference
from reduced_exchange import rollout


def digest(f):
    """Identify the exact saved linear map or compressed initial record."""
    return hashlib.sha256(f.read_bytes()).hexdigest()


def coordinates(codes, totals, profile, model, width):
    """Encode one population and return any transmitted mixture labels."""
    if "router_centers" in model:
        return encode_mixture(*profile, totals, model, width)
    if "center" in model:
        return store_geometry(*profile, totals, model, width), None
    return project(codes, totals, model, width), None


def restored(persisted, archive, model, degree):
    """Select the decoder recorded in the shared model."""
    if "router_centers" in model:
        labels = unpack_routes(
            archive["routes"], len(persisted), len(model["router_centers"])
        )
        return decode_mixture(persisted, labels, archive["totals"], model)
    if "center" in model:
        return read_geometry(persisted, archive["totals"], model)
    return decode(
        recover(persisted, archive["totals"], model), archive["totals"], degree
    )


def train(
    root,
    out,
    degree,
    widths,
    use_geometry=False,
    calibrate_int8=False,
    experts=1,
):
    """Fit all widths from a shared training matrix and its singular vectors."""
    folders = [
        root / "data/multicycle/train",
        root / "small-transfer/2ms/data/train",
    ]
    codes, totals, ids = training_records(folders, degree)
    profile = geometry(codes, totals, degree) if use_geometry else None
    if experts > 1:
        if profile is None:
            raise ValueError("Mixture routing requires conditional geometry")
        model = fit_mixture(*profile, totals, degree, experts)
    elif use_geometry:
        model = fit_geometry(*profile, totals, degree)
    else:
        model = fit(codes, totals, degree)
    axes = model["axes"].shape[-2]
    if calibrate_int8:
        full, _ = coordinates(
            codes, totals, profile, model, 2 * degree - 2 + axes
        )
        quant_scale = np.maximum(
            np.abs(full[:, 2 * degree - 2 :]).max(0) / 127, 1e-12
        )
    out.mkdir(exist_ok=True, parents=True)
    np.save(out / "train-ids.npy", ids)
    for width in widths:
        keep = width - (2 * degree - 2)
        if not 0 <= keep <= axes:
            raise ValueError(f"Unsupported width {width}")
        payload = {
            k: v
            for k, v in model.items()
            if k not in ("axes", "singular_values")
        }
        if calibrate_int8:
            payload["quant_scale"] = quant_scale[:keep]
        np.savez_compressed(
            out / f"n{degree}-w{width}.npz",
            **payload,
            axes=model["axes"][..., :keep, :],
        )
    print(len(ids), len(codes), flush=True)


def pack_record(record, model, degree, precision):
    """Keep dry moments in fp32 and round the conditional coordinates."""
    if precision == "float32":
        return dict(record=record.astype(np.float32)), 0
    split = 2 * degree - 2
    conditional = record[:, split:]
    saturated = 0
    if precision in ("int10", "int12"):
        from precision_horizon import pack

        bits = int(precision[3:])
        radius = model["quant_scale"] * 127
        packed, _ = (
            pack(conditional, 0, radius, bits)
            if conditional.size
            else (np.zeros(0, np.uint8), 0)
        )
        return dict(
            size=record[:, :split].astype(np.float32), packed=packed, bits=bits
        ), int((np.abs(conditional) > radius).sum())
    if precision == "int8":
        scaled = np.rint(conditional / model["quant_scale"])
        saturated = int((np.abs(scaled) > 127).sum())
        conditional = scaled.clip(-127, 127)
    return dict(
        size=record[:, :split].astype(np.float32),
        conditional=conditional.astype(precision),
    ), saturated


def unpack_record(archive, model):
    """Decode the actual bytes and use only training-calibrated quantization."""
    if "record" in archive:
        return archive["record"]
    if "packed" in archive:
        from precision_horizon import unpack

        radius = model["quant_scale"] * 127
        conditional = unpack(
            archive["packed"],
            0,
            radius,
            int(archive["bits"]),
            (len(archive["size"]), len(radius)),
        )
        return np.column_stack((archive["size"], conditional))
    conditional = archive["conditional"]
    if conditional.dtype == np.int8:
        conditional = conditional * model["quant_scale"]
    return np.column_stack((archive["size"], conditional))


def save_prediction(folder, name, ids, time, water, gas, record, checkpoint):
    """Retain the exact forecast emitted from a reopened compact record."""
    if folder is None:
        return
    folder.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        folder / (name + ".npz"),
        ids=ids,
        time=time,
        water=water,
        gas=gas,
        record_sha256=digest(record),
        model_sha256=digest(checkpoint),
    )


@torch.inference_mode()
def evaluate(
    root,
    models,
    archive,
    out,
    degree,
    widths,
    steps,
    precision="float32",
    mode="multicycle",
    split="validation",
    predictions=None,
):
    """Read true input archives, forecast, and score resolved observations."""
    files, ids, times, pair, bound, ref, ref_level = load_reference(
        root, mode, split
    )
    common, mass, counts, _ = inputs(root, files, ids, mode, split)
    if np.intersect1d(ids, np.load(models / "train-ids.npy")).size:
        raise ValueError("Validation families overlap training")
    truth = dict(
        time=torch.from_numpy(np.repeat(times, 2, axis=0)),
        water=torch.from_numpy(ref.reshape(-1, ref.shape[-1])),
        ref=(pair, bound),
    )
    codes, totals = encode(mass, counts, degree)
    rows = []
    profile = None
    archive.mkdir(exist_ok=True, parents=True)
    for width in widths:
        name = f"n{degree}-w{width}"
        cp = models / (name + ".npz")
        with np.load(cp) as model:
            if "center" in model and profile is None:
                profile = geometry(codes, totals, degree)
            record, labels = coordinates(codes, totals, profile, model, width)
            storage, saturated = pack_record(record, model, degree, precision)
            if labels is not None:
                storage["routes"] = pack_routes(
                    labels, len(model["router_centers"])
                )
        cp_hash = digest(cp)
        file = archive / (name + ".npz")
        np.savez_compressed(
            file, **storage, totals=totals, model_sha256=cp_hash, **common
        )
        with np.load(file) as z, np.load(cp) as model:
            if z["model_sha256"].item() != digest(cp):
                raise ValueError("Linear projection changed")
            persisted = unpack_record(z, model)
            shape, changed = restored(persisted, z, model, degree)
            weights, nodes = restore_arrays(shape, z["totals"])
            f = forcing(
                z["lengths"], z["gas_levels_kg_m3"], float(z["interval_s"])
            )
            gas = np.repeat(z["gas_levels_kg_m3"][:, 0], 2)
            args = [
                torch.from_numpy(a)
                for a in (nodes, weights / z["volume_m3"], gas, f)
            ]
        for substeps in steps:
            pred, g = rollout(*args, substeps, True)
            save_prediction(
                predictions,
                f"{name}-s{substeps}",
                ids,
                times,
                pred,
                g,
                file,
                cp,
            )
            scores = metrics_for_series(pred, g, truth, True)
            for pattern, result in scores.items():
                rows.append(
                    dict(
                        radius_nodes=degree,
                        width=width,
                        conditional_precision=precision,
                        saturated_coordinates=saturated,
                        substeps=substeps,
                        reset_times_ms=pattern,
                        projected_boxes=int(np.sum(changed > 0)),
                        archive_bytes=file.stat().st_size,
                        member_reference=ref_level,
                        shared_bytes=cp.stat().st_size,
                        archive_sha256=digest(file),
                        **result,
                    )
                )
        print(name, rows[-len(scores)]["horizon_s"], flush=True)
    with out.open("w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=rows[0], lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "evaluate"])
    p.add_argument("root", type=Path)
    p.add_argument("models", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--calibrate-int8", action="store_true")
    p.add_argument("--experts", type=int, default=1)
    p.add_argument("--geometry", action="store_true")
    p.add_argument(
        "--precision",
        choices=["float16", "float32", "int8", "int10", "int12"],
        default="float32",
    )
    p.add_argument("--degree", type=int, default=5)
    p.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[8, 9, 10, 12, 14, 16, 20, 24, 28, 33],
    )
    p.add_argument("--steps", type=int, nargs="+", default=[6])
    p.add_argument(
        "--mode", choices=["pulse", "multicycle"], default="multicycle"
    )
    p.add_argument("--split", default="validation")
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.action == "train":
        train(
            a.root,
            a.models,
            a.degree,
            a.widths,
            a.geometry,
            a.calibrate_int8,
            a.experts,
        )
    else:
        evaluate(
            a.root,
            a.models,
            a.archive,
            a.out,
            a.degree,
            a.widths,
            a.steps,
            a.precision,
            a.mode,
            a.split,
        )
