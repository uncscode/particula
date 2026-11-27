"""Encode a weighted aerosol and forecast from its compact one-vapor record."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from conditional_geometry import geometry
from moment_quadrature import encode
from multicycle_storage import forcing
from projection_study import (
    coordinates,
    digest,
    pack_record,
    restored,
    unpack_record,
)
from quadrature import restore_arrays
from reduced_exchange import rollout

DENSITY = np.array([1000.0, 1400.0, 1400.0])


def save(out, fields):
    """Publish one complete record after writing all arrays."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as f:
        np.savez_compressed(f, **fields)
    Path(f.name).replace(out)


def schedules(lengths, levels, interval, volume, boxes):
    """Accept one per-box schedule or one schedule shared by each pair."""
    if (
        levels.ndim != 2
        or lengths.shape != levels.shape
        or not len(lengths)
        or not np.isfinite(levels).all()
        or np.any(levels < 0)
        or lengths.dtype.kind not in "iu"
        or np.any(lengths <= 0)
        or not np.isfinite([interval, volume]).all()
        or min(interval, volume) <= 0
        or np.ptp(lengths.sum(1)) != 0
        or len(lengths) not in (boxes, boxes // 2)
    ):
        raise ValueError("Inconsistent gas schedule, box volume or time grid")
    if len(lengths) != boxes:
        if len(lengths) * 2 != boxes:
            raise ValueError("A shared schedule requires paired boxes")
        lengths = lengths.repeat(2, 0)
        levels = levels.repeat(2, 0)
    return forcing(lengths, levels, interval)[::2].copy(), levels[:, 0].copy()


def particle_state(d):
    """Read weighted per-particle masses in calibrated material order."""
    m, n = d["mass_kg"], d["represented_count"]
    if (
        m.shape != (*n.shape, 3)
        or np.any(n < 0)
        or np.any(m < 0)
        or not all(np.isfinite(x).all() for x in (m, n))
    ):
        raise ValueError("Invalid three-material particle masses or counts")
    if not np.array_equal(d["density"], DENSITY):
        raise ValueError("Expected water and the two calibrated dry materials")
    return m, n


def embed(raw, basis, out):
    """Pack a trained size/composition projection of this population."""
    with np.load(raw) as d, np.load(basis) as model:
        m, n = particle_state(d)
        schedules(
            d["lengths"],
            d["gas_levels_kg_m3"],
            float(d["interval_s"]),
            float(d["volume_m3"]),
            len(n),
        )
        degree = int(model["degree"])
        codes, totals = encode(m, n, degree)
        width = 2 * degree - 2 + len(model["axes"])
        profile = geometry(codes, totals, degree)
        z, _ = coordinates(codes, totals, profile, model, width)
        data, saturated = pack_record(z, model, degree, "int12")
        metadata = {
            k: d[k]
            for k in (
                "density",
                "interval_s",
                "volume_m3",
                "lengths",
                "gas_levels_kg_m3",
            )
        }
        save(
            out,
            dict(
                **metadata,
                **data,
                totals=totals,
                model_sha256=digest(basis),
                saturated_coordinates=saturated,
            ),
        )


@torch.inference_mode()
def predict(record, basis, out, steps=6):
    """Forecast both phases and check all prescribed gas resets."""
    with np.load(record) as a, np.load(basis) as model:
        if str(a["model_sha256"]) != digest(basis) or not np.array_equal(
            a["density"], DENSITY
        ):
            raise ValueError("Different compact basis or materials")
        shape, _ = restored(
            unpack_record(a, model), a, model, int(model["degree"])
        )
        n, m = restore_arrays(shape, a["totals"])
        vol = float(a["volume_m3"])
        f, g = schedules(
            a["lengths"],
            a["gas_levels_kg_m3"],
            float(a["interval_s"]),
            vol,
            len(n),
        )
        n = n / vol
    nw, ng = rollout(
        *map(torch.from_numpy, (m, n, g, f)), steps, True, return_nodes=True
    )
    node = nw.numpy()
    gas = ng.numpy()
    wet = (node * n[:, None]).sum(-1)
    dry = m[..., 1:].sum(-1)
    diameter = 2e6 * np.cbrt(
        3 / (4 * np.pi) * (dry[:, None] / 1400 + node / 1000)
    )
    edges = np.geomspace(0.08, 8.0, 13)[1:-1]
    k = np.searchsorted(edges, diameter)
    count = np.stack([((k == j) * n[:, None]).sum(-1) for j in range(12)], -1)
    time = np.c_[np.zeros(len(n)), np.cumsum(f[..., 0], 1)]
    mask = f[..., 3] > 0
    change = np.diff(wet + gas, axis=1)
    np.testing.assert_allclose(change[~mask], 0, atol=2e-14)
    np.testing.assert_allclose(gas[:, 1:][mask], f[..., 4][mask], atol=1e-16)
    np.testing.assert_allclose(np.diff(wet, axis=1)[mask], 0, atol=1e-16)
    if not np.isfinite(wet + gas).all() or min(wet.min(), gas.min()) < 0:
        raise ArithmeticError("Nonphysical forecast")
    save(
        out,
        dict(
            time_s=time,
            particle_water_kg_m3=wet,
            gas_water_kg_m3=gas,
            wet_size_number_m3=count,
            interior_diameter_edges_um=edges,
            initial_state_sha256=digest(record),
            basis_sha256=digest(basis),
            substeps=steps,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    opts = p.add_subparsers(dest="action", required=True)
    for action, first in (("encode", "raw"), ("forecast", "record")):
        a = opts.add_parser(action)
        for f in (first, "basis", "out"):
            a.add_argument(f, type=Path)
        if action == "forecast":
            a.add_argument("--steps", type=int, default=6)
    d = vars(p.parse_args())
    if d["out"].resolve() in [
        v.resolve() for k, v in d.items() if k != "out" and isinstance(v, Path)
    ]:
        p.error("Choose a distinct output path")
    torch.set_num_threads(1)
    {"encode": embed, "forecast": predict}[d.pop("action")](**d)
