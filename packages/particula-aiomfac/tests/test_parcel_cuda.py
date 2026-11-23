"""Native device parcel rollback and restart identity on a real CUDA device."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("particula")
pytestmark = pytest.mark.skipif(
    not wp.is_cuda_available(), reason="requires NVIDIA CUDA"
)

from particula.dynamics.multivapor_parcel import (
    MultiVaporParcelInitialState,
    ParcelVapor,
)
from particula.dynamics.parcel import ParcelRK23, ParcelROS3, run_parcel
from particula.execution.multivapor_parcel import (
    ParcelReplayCheckpoint,
    ResidentMultiVaporParcel,
)
from particula.particles.activity_factories import ActivityFactory
from particula_aiomfac import MixtureSpec


def make_state(case_name):
    folder = Path(__file__).resolve().parents[1] / "reference/cases"
    case = json.loads((folder / f"{case_name}.json").read_text())
    counts = json.loads((folder / "components.json").read_text())
    groups = {
        name: {int(i): n for i, n in counts[name].items()}
        for name in case["components"]
    }
    activity = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(groups),
            material_mapping=dict(enumerate(groups)),
        ),
    )
    fractions = np.array(case["fractions"])
    return MultiVaporParcelInitialState(
        masses=np.array([fractions * 5e-15, fractions * 0.0]),
        number=np.array([1e6, 0.0]),
        vapor=np.array([0.022]),
        material_names=tuple(groups),
        density=np.full(len(groups), 1000.0),
        vapors=(
            ParcelVapor(
                "water",
                list(groups).index("water"),
                0.01801528,
                3169.0,
                298.15,
                2.5e6,
                2.11e-5,
            ),
        ),
        activity_strategy=activity,
        pressure=101325.0,
        temperature=case["temperature_K"],
    )


@pytest.mark.parametrize(
    ("control", "hybrid"),
    [(ParcelRK23(), False), (ParcelROS3(), False), (ParcelROS3(), True)],
)
def test_backend_failure_preserves_accepted_coordinates(
    control, hybrid, tmp_path
):
    s = make_state("coupled-trace-bound")
    if hybrid:
        from particula_aiomfac.sparse import TableRecipe, build_table

        provider = s.activity_strategy
        recipe = TableRecipe(
            name="coupled",
            components=tuple(provider.mixture.components.items()),
            dry_weights=(0, 1, 1),
            targets=("water",),
            variable_dry=(),
            bounds=((3.0, 3.1), (295.0, 296.0)),
            log_tolerances=(4e-5,),
            max_level=1,
        )
        table = tmp_path / "coupled.npz"
        build_table(recipe).save(table)
        s = replace(
            s,
            activity_strategy=ActivityFactory().get_strategy(
                "aiomfac",
                dict(
                    mixture=provider.mixture,
                    material_mapping=dict(enumerate(s.material_names)),
                    evaluation="hybrid",
                    table=table,
                ),
            ),
        )
    owner = ResidentMultiVaporParcel(
        s, device="cuda:0", adaptive=control, timestep=1e-3, updraft=1.0
    )
    capture = owner.checkpoint()
    with pytest.raises(
        RuntimeError, match=r"box 0, slot 0, solver stage 1, backend status -"
    ):
        owner.advance(0.001)
    np.testing.assert_array_equal(owner.y.numpy(), capture.coordinates)
    assert owner.clock.numpy()[0] == 0
    if hybrid:
        assert owner.binding.counts.numpy()[6] > 0


@pytest.mark.parametrize("control", [ParcelRK23(), ParcelROS3()])
def test_native_entry_and_incompatible_restart(control, tmp_path):
    state = make_state("mixed")
    kwargs = dict(device="cuda:0", adaptive=control, timestep=0.02, updraft=0.5)
    native = run_parcel(state, steps=2, **kwargs)
    cpu = run_parcel(state, steps=2, timestep=0.02, updraft=0.5)
    np.testing.assert_allclose(
        native.temperature, cpu.temperature, rtol=0, atol=0.01
    )
    np.testing.assert_allclose(native.wet_radius, cpu.wet_radius, rtol=1e-3)
    owner = ResidentMultiVaporParcel(state, **kwargs)
    before = owner.checkpoint()
    owner.advance(0.02)
    now = owner.checkpoint()
    for corrupt in (
        replace(before, identity=dict(before.identity, updraft=0)),
        replace(before, vapor=np.zeros(1)),
    ):
        with pytest.raises(ValueError, match="incompatible|invalid"):
            owner.restore(corrupt)
        np.testing.assert_array_equal(
            now.coordinates, owner.checkpoint().coordinates
        )
    archive = tmp_path / "parcel.zip"
    before.save(archive)
    owner.restore(ParcelReplayCheckpoint.load(archive))
    owner.advance(0.02)
    np.testing.assert_array_equal(
        now.coordinates, owner.checkpoint().coordinates
    )
