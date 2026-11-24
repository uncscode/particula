"""Verify the independent OpenFOAM fixture, transfer mapping and replay."""

import json

import numpy as np
import pytest
import warp as wp
from particula.coupling.cavity_replay import map_cavity, read_cavity
from particula.coupling.grid_inventory import InventoryLayout, Uptake
from particula.coupling.grid_run import GridRun
from particula_docs import documentation_path

DATA = documentation_path() / "data/prescribed-flow-cavity"


@pytest.fixture
def export():
    """Read the original v2312 arrays in native order."""
    with np.load(DATA / "cavity.npz") as archive:
        return json.loads((DATA / "cavity.json").read_text()), dict(archive)


def test_native_identity_and_orientation(export):
    """Renumber and permute cells and faces, reversing half the internal normals."""
    meta, data = export
    ref = map_cavity(meta, data)
    rng = np.random.default_rng(72)
    row = rng.permutation(len(data["cell_id"]))
    rename = 100 + 3 * rng.permutation(len(row))
    data["cell_id"] = rename[data["cell_id"]]
    data["owner"] = rename[data["owner"]]
    inside = data["neighbour"] >= 0
    data["neighbour"][inside] = rename[data["neighbour"][inside]]
    for key in (
        "cell_id",
        "centres",
        "volumes",
        "velocity",
        "kinematic_pressure",
    ):
        data[key] = data[key][row]
    for key in ("scalar", "half_step", "fine_restricted"):
        data[key] = data[key][:, row]
    flip = np.nonzero(inside)[0][::2]
    data["owner"][flip], data["neighbour"][flip] = (
        data["neighbour"][flip],
        data["owner"][flip],
    )
    data["area_vectors"][flip] *= -1
    data["phi"][flip] *= -1
    order = rng.permutation(len(inside))
    for key in (
        "owner",
        "neighbour",
        "patch",
        "phi",
        "area_vectors",
        "face_centres",
    ):
        data[key] = data[key][order]
    result = map_cavity(meta, data)
    np.testing.assert_array_equal(result.concentration, ref.concentration)
    for a, b in zip(result.forcing.frames, ref.forcing.frames, strict=True):
        np.testing.assert_array_equal(a.transfer, b.transfer)
        np.testing.assert_array_equal(a.mixing, b.mixing)


@pytest.mark.parametrize(
    "fault",
    [
        "units",
        "missing",
        "area",
        "orientation",
        "rate",
        "patch",
        "duplicate",
        "dt",
    ],
)
def test_malformed_native_export(export, fault):
    """Reject incomplete records, geometry, flux signs and ambiguous units."""
    meta, data = export
    if fault == "units":
        meta["units"]["phi"] = "kg/s"
    elif fault == "missing":
        data["step"] = np.delete(data["step"], 5)
    elif fault == "area":
        data["area_vectors"][2] *= 0.5
    elif fault == "orientation":
        data["area_vectors"][2] *= -1
    elif fault == "rate":
        data["phi"][2] *= -1
    elif fault == "patch":
        data["patch"][-1] = 0
    elif fault == "duplicate":
        for key in (
            "owner",
            "neighbour",
            "patch",
            "phi",
            "area_vectors",
            "face_centres",
        ):
            data[key][2] = data[key][3]
    elif fault == "dt":
        data["delta_t"][0] *= 2
    with pytest.raises(ValueError):
        map_cavity(meta, data)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_replay_refinement_restart_uptake(device, tmp_path):
    """Compare each device to the host at every record, including a restart."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("NVIDIA CUDA unavailable")
    case = read_cavity(DATA)
    forcing = case.forcing
    grid = forcing.grid
    initial = grid.volume * case.concentration[0]
    layout = InventoryLayout(("tracer",))
    run = GridRun(forcing, layout, initial[:, None], device=device)
    active_layout = InventoryLayout(
        ("tracer",), ("seed",), ("core", "condensed")
    )
    active = GridRun(
        forcing,
        active_layout,
        np.column_stack((initial, 100 * grid.volume, grid.volume, 0 * initial)),
        uptake=Uptake("tracer", "seed", "condensed", 8),
        device=device,
    )
    limits = case.metadata["limits"]
    midpoint = len(forcing.frames) // 2
    restarted = None
    for i, frame in enumerate(forcing.frames, start=1):
        run.advance()
        active.advance()
        mass = run.snapshot()["state"][:, 0]
        host = grid.volume * case.concentration[i]
        np.testing.assert_allclose(
            [mass.sum(), host.sum()],
            initial.sum(),
            rtol=limits["relative_mass"],
            atol=0,
        )
        assert abs(mass - host).sum() / initial.sum() < limits["relative_l1"]
        delta = (
            mass @ grid.centers / mass.sum() - host @ grid.centers / host.sum()
        )
        assert np.linalg.norm(delta) < limits["centroid_m"]
        budget = active.budget()["state"]
        np.testing.assert_allclose(
            budget[[0, 3]].sum(), initial.sum(), rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            budget[0] / initial.sum(),
            np.exp(-8 * (frame.end - forcing.times[0])),
            rtol=1e-12,
        )
        if i == midpoint:
            path = tmp_path / "restart.npz"
            run.checkpoint(path)
            restarted = GridRun(
                forcing, layout, initial[:, None], device=device
            )
            restarted.restore(path)
        elif i > midpoint:
            restarted.advance()
            np.testing.assert_array_equal(
                restarted.snapshot()["state"][:, 0], mass
            )
