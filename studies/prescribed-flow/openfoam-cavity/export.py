"""Export and score independent host refinements without importing Particula."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from prepare import CASES, IMAGE


def foam_text(path):
    """Strip banner, file header and comments from the native ASCII format."""
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
    return re.sub(r"FoamFile\s*\{[^}]*\}", "", text).strip()


def numbers(text, dtype=float):
    """Parse a flat sequence of ASCII scalars or labels."""
    return np.fromstring(re.sub(r"[()]", " ", text), dtype=dtype, sep=" ")


def native_list(path):
    """Read the declared list count and native entries."""
    text = foam_text(path)
    match = re.fullmatch(r"(\d+)\s*\((.*)\)", text, re.S)
    return int(match[1]), match[2]


def field(path, dimensions, width=1):
    """Read a dimension-checked field in native cell or face order."""
    text = foam_text(path)
    if (
        re.search(r"dimensions\s*\[([^\]]+)\]", text)[1].split()
        != dimensions.split()
    ):
        raise ValueError(f"unexpected dimensions: {path}")
    match = re.search(
        r"internalField\s+nonuniform List<\w+>\s*(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    values = numbers(match[2]).reshape(int(match[1]), width)
    return values[:, 0] if width == 1 else values


def mesh(case):
    """Recover owner-outward quadrilateral geometry and native addressing."""
    poly = case / "constant/polyMesh"
    _, vertices = native_list(poly / "points")
    vertices = numbers(vertices).reshape(-1, 3)
    nfaces, face_data = native_list(poly / "faces")
    quads = re.findall(r"4\(([^()]*)\)", face_data)
    if len(quads) != nfaces:
        raise ValueError("expected quadrilateral cavity faces")
    corner = vertices[np.stack([numbers(q, int) for q in quads])]
    area = 0.5 * np.cross(
        corner[:, 2] - corner[:, 0], corner[:, 3] - corner[:, 1]
    )
    owner = numbers(native_list(poly / "owner")[1], int)
    inner = numbers(native_list(poly / "neighbour")[1], int)
    neighbour = np.full(nfaces, -1)
    neighbour[: len(inner)] = inner
    patch = np.zeros(nfaces, dtype=np.int8)
    boundary = foam_text(poly / "boundary")
    for code, name in enumerate(("lid", "walls", "frontBack"), 1):
        block = re.search(name + r"\s*\{([^}]+)\}", boundary)[1]
        start = int(re.search(r"startFace\s+(\d+)", block)[1])
        count = int(re.search(r"nFaces\s+(\d+)", block)[1])
        patch[start : start + count] = code
    return dict(
        owner=owner,
        neighbour=neighbour,
        patch=patch,
        face_centres=corner.mean(axis=1),
        area_vectors=area,
    )


def load(case):
    """Collect native fields and confirm frozen fluxes across all writes."""
    times = sorted(
        (p for p in case.iterdir() if re.fullmatch(r"0\.\d+", p.name)),
        key=lambda p: float(p.name),
    )
    start = times[0]
    geom = mesh(case)
    interior = geom["neighbour"] >= 0
    phi = np.zeros(len(interior))
    phi[interior] = field(start / "phi", "0 3 -1 0 0 0 0")
    flow = field(start / "U", "0 1 -1 0 0 0 0", 3)
    scalar, indices, dt = [], [], []
    for p in times:
        scalar.append(field(p / "T", "1 -3 0 0 0 0 0"))
        # scalarTransportFoam reuses the saved pressure-corrected phi exactly.
        np.testing.assert_array_equal(
            field(p / "phi", "0 3 -1 0 0 0 0"), phi[interior]
        )
        np.testing.assert_array_equal(field(p / "U", "0 1 -1 0 0 0 0", 3), flow)
        flux_text = foam_text(p / "phi")
        for boundary in ("lid", "walls"):
            entry = re.search(boundary + r"\s*\{([^}]+)\}", flux_text)[1]
            if not re.search(r"value\s+uniform\s+0;", entry):
                raise ValueError("nonzero or unsupported wall flux")
        if not re.search(r"frontBack\s*\{\s*type\s+empty;", flux_text):
            raise ValueError("missing empty patch")
        step = foam_text(p / "uniform/time")
        indices.append(int(re.search(r"index\s+(\d+)", step)[1]))
        dt.append(float(re.search(r"deltaT\s+([^;]+)", step)[1]))
    time = np.array([float(p.name) for p in times])
    np.testing.assert_array_equal(np.diff(indices), 1)
    np.testing.assert_allclose(np.diff(time), dt[1:], atol=2e-15, rtol=0)
    centres = field(start / "C", "0 1 0 0 0 0 0", 3)
    return dict(
        **geom,
        cell_id=np.arange(len(centres)),
        centres=centres,
        volumes=field(start / "V", "0 3 0 0 0 0 0"),
        velocity=flow,
        phi=phi,
        scalar=np.array(scalar),
        time=time,
        step=np.asarray(indices),
        delta_t=np.asarray(dt[1:]),
        kinematic_pressure=field(start / "p", "0 2 -2 0 0 0 0"),
    )


def restrict(reference, fine, n):
    """Restrict by physical location and native cell volume."""

    def bins(case):
        xy = np.floor(case["centres"][:, :2] * (n / 0.1)).astype(int)
        return np.ravel_multi_index(xy.T, (n, n))

    parent = bins(fine)
    index = np.searchsorted(fine["time"], reference["time"])
    np.testing.assert_array_equal(fine["time"][index], reference["time"])
    volumes = np.bincount(parent, weights=fine["volumes"])
    coarse_ids = bins(reference)
    np.testing.assert_allclose(volumes[coarse_ids], reference["volumes"])
    return np.array(
        [
            np.bincount(parent, weights=row * fine["volumes"])[coarse_ids]
            / volumes[coarse_ids]
            for row in fine["scalar"][index]
        ]
    )


def export(root, target):
    """Freeze the host-only refinement envelope and write the archive."""
    target.mkdir(parents=True, exist_ok=True)
    ref, half, fine = [load(root / name) for name in CASES]
    ref["half_step"] = restrict(ref, half, 16)
    ref["fine_restricted"] = restrict(ref, fine, 16)
    vol = ref["volumes"]
    initial = np.sum(ref["scalar"][0] * vol)

    def errors(a, b):
        l1 = np.sum(abs(a - b) * vol, axis=1) / initial

        def centroid(field):
            mass = field * vol
            return (mass @ ref["centres"]) / mass.sum(axis=1)[:, None]

        disp = np.linalg.norm(centroid(a) - centroid(b), axis=1)
        return max(l1), max(disp)

    temporal = errors(ref["scalar"], ref["half_step"])
    spatial = errors(ref["half_step"], ref["fine_restricted"])
    path = target / "cavity.npz"
    np.savez_compressed(path, **ref)
    meta = {
        "schema": "particula-openfoam-cavity-v1",
        "image": IMAGE,
        "build": "OpenCFD v2312 c39a0f64-20231220; linux64GccDPInt32Opt",
        "solver": "icoFoam, then scalarTransportFoam with frozen phi and U",
        "rights": (
            "Original case inputs and numerical output distributed "
            "under the repository MIT license."
        ),
        "shape": [16, 16, 1],
        "bounds": [[0, 0.1], [0, 0.1], [0, 0.01]],
        "first_step": int(ref["step"][0]),
        "last_step": int(ref["step"][-1]),
        "units": {
            "time": "s",
            "centres": "m",
            "volumes": "m3",
            "face_centres": "m",
            "area_vectors": "m2",
            "phi": "m3/s",
            "velocity": "m/s",
            "scalar": "kg/m3",
            "kinematic_pressure": "m2/s2",
        },
        "patches": ["internal", "lid", "walls", "frontBack"],
        "phi_convention": (
            "constant over each accepted interval, positive out of owner"
        ),
        "diffusivity_m2_s": 1e-5,
        "temperature_K": 293.15,
        "pressure_Pa": 101325.0,
        "vapor_kg_per_kg_dry_air": 0.005,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "refinement": {
            "half_step_l1": temporal[0],
            "half_step_centroid_m": temporal[1],
            "fine_mesh_l1": spatial[0],
            "fine_mesh_centroid_m": spatial[1],
        },
        "limits": {
            "relative_l1": max(5 * temporal[0], spatial[0]),
            "centroid_m": max(5 * temporal[1], spatial[1]),
            "relative_mass": 1e-9,
        },
    }
    (target / "cavity.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta["refinement"], meta["limits"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    export(args.directory, args.target)
