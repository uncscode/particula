"""Mass closure and analytical controls for finite-bed dust emission."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp
from particula.coupling.flow_replay import GridReplay
from particula.coupling.grid import FlowGrid
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_run import GridRun
from particula.coupling.prescribed import Sounding
from particula.coupling.sounding_sequence import (
    ChangingSoundingFlow,
    SoundingSequence,
)
from particula.coupling.surface_emission import (
    BedPatch,
    DustClass,
    SurfaceEmission,
)

GRID = FlowGrid(
    (1, 1, 1),
    ((0, 10), (0, 1), (0, 5)),
    ("periodic", "periodic", *("closed",) * 4),
)
LAYOUT = InventoryLayout((), ("fine", "coarse"), ("silica", "oxide"))
CLASSES = (
    DustClass("fine", 1.0, 1.0, 2e-6, 2500.0, (0.8, 0.2)),
    DustClass("coarse", 1.0, 1.0, 8e-6, 2500.0, (0.3, 0.7)),
)
PATCH = BedPatch("crest", 0, 2.0, (2.0, 10.0))


def setup(winds, times, *, north=0.0, device="cpu"):
    """Prescribe exact linear winds between knots in a uniform column."""
    profiles = tuple(
        Sounding(
            [0.0, 5.0],
            [288.0, 288.0],
            [1e5, 0.99e5],
            [0.01, 0.01],
            [w, w],
            [north, north],
        )
        for w in winds
    )
    flow = ChangingSoundingFlow(
        SoundingSequence(
            tuple(times), profiles, "0 s", "synthetic uniform column"
        )
    )
    return GridReplay.record(
        GRID, flow, times, device=device, provenance="synthetic"
    ), flow


def source(
    forcing, flow, mode, *, patches=(PATCH,), device="cpu", capacity=None
):
    """Use Cd=1 to let prescribed wind equal friction velocity."""
    return SurfaceEmission(
        forcing,
        flow,
        patches,
        CLASSES,
        LAYOUT.materials,
        reference_height=0.5,
        ejection_height=0.25,
        drag_coefficient=1.0,
        representation=mode,
        capacity=capacity,
        device=device,
    )


@pytest.mark.parametrize("mode", ["binned", "resolved"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_two_gust_inventory_and_restart(mode, device, tmp_path):
    """One class exhausts in gust one; each mode closes against the same dry kg."""
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    forcing, flow = setup(
        [0.0, 2.0, 0.0, 2.0, 0.0], np.arange(5.0), device=device
    )
    surf = source(forcing, flow, mode, device=device)
    # Integral from 1/2 to 1 of (4t²-1) dt = 2/3; multiply by area 2.
    expected = np.array([[2.0, 16.0 / 3.0]])

    def fresh():
        return GridRun(
            forcing,
            LAYOUT,
            np.zeros((1, LAYOUT.width)),
            surface=surf,
            device=device,
        )

    run = fresh()
    run.advance()
    path = tmp_path / "restart.npz"
    run.checkpoint(path)
    clone = fresh()
    clone.restore(path)
    while run.index < len(forcing.frames):
        run.advance()
        clone.advance()
    for key, arr in run.snapshot().items():
        np.testing.assert_array_equal(arr, clone.snapshot()[key])
    out = run.budget()
    np.testing.assert_allclose(out["surface_emitted_kg"], expected)
    np.testing.assert_allclose(
        out["surface_bed_kg"], np.array(PATCH.available) - expected
    )
    np.testing.assert_allclose(surf.emitted[2:, 0, 0], 0.0)
    kg = expected[0, :, None] * np.array([d.fractions for d in CLASSES])
    counts = expected[0] / np.array([d.mass for d in CLASSES])
    if mode == "binned":
        np.testing.assert_allclose(out["state"][:2], counts)
        np.testing.assert_allclose(out["state"][2:].reshape(2, 2), kg)
    else:
        np.testing.assert_allclose(
            out["airborne"], [counts.sum(), 0.0, *kg.sum(axis=0)]
        )
        for j in range(2):
            records = surf.records[:, 2] == j
            np.testing.assert_allclose(
                run.payload[records].sum(axis=0), [counts[j], 0.0, *kg[j]]
            )
    np.testing.assert_allclose(
        out["bed_mass"] + kg.sum(axis=0),
        np.asarray(PATCH.available)
        @ np.asarray([d.fractions for d in CLASSES]),
    )
    # Corrupting a patch's inventory must fail before changing the run.
    with np.load(path) as data:
        saved = dict(data)
    saved["surface_bed_kg"] += 1.0
    np.savez(path, **saved)
    with pytest.raises(ValueError, match="source phase"):
        clone.restore(path)


@pytest.mark.parametrize(
    ("winds", "mass"),
    [([0.5, 0.5], 0.0), ([2.0, 2.0], 6.0), ([-2.0, 2.0], 4.0 / 3.0)],
)
def test_crossings_and_constant_control(winds, mass):
    """Exercise below-threshold, constant and two-sided threshold crossings."""
    forcing, flow = setup(winds, [0.0, 1.0])
    patch = replace(PATCH, available=(20.0, 20.0))
    surf = source(forcing, flow, "binned", patches=(patch,))
    np.testing.assert_allclose(surf.emitted.sum(axis=0), [[mass, mass]])


def test_capacity_and_multiple_patches():
    """Preflight capacity across the full schedule; allow shared bottom cells."""
    forcing, flow = setup([2.0, 2.0], [0.0, 1.0])
    patches = (PATCH, replace(PATCH, name="toe", area=0.5))
    with pytest.raises(ValueError, match="capacity"):
        source(forcing, flow, "resolved", patches=patches, capacity=3)
    surf = source(forcing, flow, "resolved", patches=patches, capacity=4)
    assert len({r.particle.id for r in surf.bind(forcing, LAYOUT)}) == 4
    other = source(forcing, flow, "binned", patches=patches)
    run = GridRun(forcing, LAYOUT, np.zeros((1, LAYOUT.width)), surface=other)
    run.advance()
    np.testing.assert_allclose(
        run.budget()["state"][2:].reshape(2, 2).sum(axis=1), [3.5, 7.5]
    )


def test_shear_tangent_at_midpoint():
    """The wind norm touches threshold once, with positive flux on both sides."""
    forcing, flow = setup([-1.0, 1.0], [0.0, 1.0], north=1.0)
    coarse, _ = setup([0.0, 0.0], [0.0, 1.0])
    for history in (forcing, coarse):
        surf = source(history, flow, "binned")
        np.testing.assert_allclose(surf.emitted.sum(axis=0), 2.0 / 3.0)


def test_internal_wind_knot_and_patch_area():
    """Preserve a triangular gust when transport has only one long interval."""
    steady, _ = setup([0.0, 0.0], [0.0, 2.0])
    _, gust = setup([0.0, 2.0, 0.0], [0.0, 1.0, 2.0])
    surf = source(steady, gust, "binned")
    np.testing.assert_allclose(surf.emitted[0, 0], [2.0, 8.0 / 3.0])
    with pytest.raises(ValueError, match="area exceeds"):
        source(steady, gust, "binned", patches=(replace(PATCH, area=11.0),))
