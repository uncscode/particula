"""Map a stationary Cartesian OpenFOAM cavity export to SI forcing."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from particula.coupling.flow_replay import GridReplay
from particula.coupling.flows import GridInterval
from particula.coupling.grid import FlowGrid

_UNITS = {
    "time": "s",
    "centres": "m",
    "volumes": "m3",
    "face_centres": "m",
    "area_vectors": "m2",
    "phi": "m3/s",
    "velocity": "m/s",
    "scalar": "kg/m3",
    "kinematic_pressure": "m2/s2",
}


@dataclass(frozen=True)
class CavityReference:
    """Forcing and independent scalar histories (kg/m³) in grid row order."""

    forcing: GridReplay
    concentration: np.ndarray
    half_step: np.ndarray
    fine_restricted: np.ndarray
    metadata: dict


def read_cavity(directory):
    """Verify the exported archive SHA-256 before mapping native cell IDs."""
    directory = Path(directory)
    metadata = json.loads((directory / "cavity.json").read_text())
    archive = directory / "cavity.npz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("cavity fixture checksum mismatch")
    with np.load(archive, allow_pickle=False) as arrays:
        return map_cavity(metadata, dict(arrays))


def _array(data, key, shape, *, integer=False):
    value = np.asarray(data[key])
    if (
        value.shape != shape
        or not np.issubdtype(value.dtype, np.number)
        or not np.isfinite(value).all()
        or (integer and not np.issubdtype(value.dtype, np.integer))
    ):
        raise ValueError(f"invalid {key} dimensions or values")
    return value


def _cell_mapping(grid, data):
    n = len(grid.volume)
    ids = _array(data, "cell_id", (n,), integer=True)
    if np.any(ids < 0) or len(np.unique(ids)) != n:
        raise ValueError("native cell IDs must be unique and nonnegative")
    centres = _array(data, "centres", (n, 3))
    indices = np.floor(
        (centres - np.array(grid.bounds)[:, 0]) / grid.spacing
    ).astype(int)
    ix = np.ravel_multi_index(indices.T, grid.shape)
    if (
        len(np.unique(ix)) != n
        or not np.allclose(centres, grid.centers[ix], atol=1e-12, rtol=0)
        or not np.allclose(
            _array(data, "volumes", (n,)), grid.volume[ix], atol=0, rtol=1e-11
        )
    ):
        raise ValueError(
            "native cell centres or volumes disagree with the Cartesian mesh"
        )
    return dict(zip(ids, ix, strict=True)), np.argsort(ix)


def _face_rates(grid, data, lookup, diffusivity):
    """Require a single owner-outward face for every logical cell side."""
    nf = len(data["owner"])
    owner_id = _array(data, "owner", (nf,), integer=True)
    nb_id = _array(data, "neighbour", (nf,), integer=True)
    inside = nb_id != -1
    try:
        owner = np.array([lookup[x] for x in owner_id], dtype=int)
        nb = np.array([lookup[x] for x in nb_id[inside]], dtype=int)
    except KeyError as error:
        raise ValueError("unknown native face cell ID") from error
    area = _array(data, "area_vectors", (nf, 3))
    centre = _array(data, "face_centres", (nf, 3))
    axis = abs(area).argmax(axis=1)
    normal = area[np.arange(nf), axis]
    side = 2 * axis + (normal > 0)
    delta = np.eye(3)[axis] * np.sign(normal)[:, None]
    expected_area = grid.volume[owner] / grid.spacing[axis]
    if not np.allclose(
        area, delta * expected_area[:, None], rtol=1e-10, atol=1e-16
    ) or not np.allclose(
        centre,
        grid.centers[owner] + 0.5 * delta * grid.spacing,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            "face location or owner-outward orientation disagrees with geometry"
        )
    neighbour = grid.neighbors[owner, side]
    if not np.array_equal(inside, neighbour >= 0) or not np.array_equal(
        neighbour[inside], nb
    ):
        raise ValueError(
            "native face neighbour disagrees with the Cartesian mesh"
        )
    patch = _array(data, "patch", (nf,), integer=True)
    expected_patch = np.where(
        inside, 0, np.where(side == 3, 1, np.where(axis == 2, 3, 2))
    )
    if not np.array_equal(patch, expected_patch):
        raise ValueError("native cavity patch disagrees with face location")
    slots = np.r_[owner * 6 + side, nb * 6 + (side[inside] ^ 1)]
    if not np.array_equal(np.sort(slots), np.arange(6 * len(grid.volume))):
        raise ValueError("missing or duplicated native face")
    phi = _array(data, "phi", (nf,))
    if np.any(phi[~inside]):
        raise ValueError("impermeable cavity boundary has a nonzero flux")
    rates = np.zeros_like(grid.neighbors, dtype=float)
    rates[owner, side] = phi
    rates[nb, side[inside] ^ 1] = -phi[inside]
    if np.max(abs(rates.sum(axis=1))) > 1e-10 * np.max(abs(rates)):
        raise ValueError("native face rates violate cell volume continuity")
    exchange = np.zeros_like(rates)
    k = diffusivity * expected_area[inside] / grid.spacing[axis[inside]]
    exchange[owner[inside], side[inside]] = k
    exchange[nb, side[inside] ^ 1] = k
    return rates, exchange


def map_cavity(metadata, data):
    """Map every native scalar step in a stationary cavity export.

    Cell IDs may be permuted. Internal face owner/neighbour may be exchanged
    if the area vector and signed flux are reversed together. This adapter
    uses volume-based concentrations and frozen pressure-corrected face rates.
    """
    if (
        metadata["schema"] != "particula-openfoam-cavity-v1"
        or metadata["units"] != _UNITS
        or metadata["patches"] != ["internal", "lid", "walls", "frontBack"]
        or metadata["phi_convention"]
        != "constant over each accepted interval, positive out of owner"
    ):
        raise ValueError(
            "unsupported cavity schema, units, patches or flux convention"
        )
    grid = FlowGrid(tuple(metadata["shape"]), tuple(metadata["bounds"]))
    if grid.shape[2] != 1:
        raise ValueError(
            "this cavity adapter requires a one-cell physical thickness"
        )
    ids, order = _cell_mapping(grid, data)
    n = len(order)
    steps = np.arange(metadata["first_step"], metadata["last_step"] + 1)
    nt = len(steps)
    if nt < 2 or not np.array_equal(
        _array(data, "step", (nt,), integer=True), steps
    ):
        raise ValueError("incomplete accepted step history")
    times = _array(data, "time", (nt,))
    dt = np.diff(times)
    if np.any(dt <= 0) or not np.allclose(
        dt, _array(data, "delta_t", (nt - 1,)), atol=1e-15, rtol=1e-12
    ):
        raise ValueError("accepted times and native intervals disagree")
    diffusivity = metadata["diffusivity_m2_s"]
    if not np.isfinite(diffusivity) or diffusivity < 0:
        raise ValueError("invalid constant scalar diffusivity")
    rate, exchange = _face_rates(grid, data, ids, diffusivity)
    velocity = _array(data, "velocity", (n, 3))[order]
    _array(data, "kinematic_pressure", (n,))
    air = np.empty((n, 6))
    air[:, :3] = velocity
    air[:, 3:] = [
        metadata["temperature_K"],
        metadata["pressure_Pa"],
        metadata["vapor_kg_per_kg_dry_air"],
    ]
    diffusivities = np.full_like(velocity, diffusivity)
    frames = [
        GridInterval(
            times[i],
            times[i + 1],
            air,
            rate * d,
            exchange * d,
            velocity,
            diffusivities,
        )
        for i, d in enumerate(dt)
    ]
    reference = [
        _array(data, key, (nt, n))[:, order]
        for key in ("scalar", "half_step", "fine_restricted")
    ]
    if any(np.any(row < 0) for row in reference):
        raise ValueError("negative host concentration")
    return CavityReference(
        GridReplay(grid, frames, metadata["solver"]), *reference, metadata
    )
