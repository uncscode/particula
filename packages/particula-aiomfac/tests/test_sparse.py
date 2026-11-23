"""Independent neutral interpolation, admission and content identity checks."""

from dataclasses import replace

import numpy as np
import pytest
from particula_aiomfac.sparse import (
    ActivityTable,
    NodeEvaluator,
    build_table,
    load_recipe,
)


def test_reference_and_content_hash(tmp_path):
    t = build_table("mixed-water")
    path = tmp_path / "water.npz"
    t.save(path)
    model = ActivityTable.load(path).prepare()
    m = np.array([[0.977, 0.001, 0.002, 0.020]])
    a, e, reason = model.evaluate(m, 298.0)
    assert reason.item() == 0
    assert abs(a.item() - 0.9878567852129456) < 1e-4
    assert abs(np.log(a.item() / 0.9878567852129456)) <= e.item()
    with np.load(path) as src:
        data = dict(src)
    data["coefficients"][0] += 1.0
    np.savez(path, **data)
    with pytest.raises(ValueError, match="content hash"):
        ActivityTable.load(path)


@pytest.mark.parametrize(
    "recipe",
    [
        "mixed-water",
        "mixed-two-vapor",
        "mixed-variable-acid",
        "neutral",
        "shared-ion",
        "sulfate",
        "carbonate",
    ],
)
def test_independent_rows(recipe):
    t = ActivityTable.load(recipe)
    r = t.recipe
    u = np.random.default_rng(391).uniform(size=(32, len(r.bounds)))
    m, tk = r.states(u)
    truth, code = NodeEvaluator(r, "cpu").evaluate(m, tk)
    a, e, reason = t.prepare().evaluate(m, tk)
    assert np.all(code >= 0) and np.all(reason == 0)
    assert np.max(abs(truth[:, 0] - a[:, 0])) < 1e-4
    assert np.all(abs(np.log(truth / a)) < np.array(r.log_tolerances))
    assert np.all(abs(np.log(truth / a)) < e)


def test_row_fallback_and_index():
    t = ActivityTable.load("mixed-two-vapor")
    m, tk = t.recipe.states(np.full((5, 3), 0.5))
    m[1, 1] *= 2
    m[2, 2] = 0
    tk[3] = 315
    m[4] = 0
    values, _, why = t.prepare().evaluate(m, tk, active=[1, 1, 1, 1, 0])
    np.testing.assert_array_equal(why, [0, 2, 1, 3, -1])
    direct, _ = NodeEvaluator(t.recipe, "cpu").evaluate(m[1:4], tk[1:4])
    np.testing.assert_allclose(values[1:4], direct, atol=1e-14, rtol=0.0)
    with pytest.raises(ValueError, match="batch index .4,"):
        t.prepare().evaluate(m, tk)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_positive_trace_abundances_use_direct_on_simulation_device(device):
    if device != "cpu" and not pytest.importorskip("warp").is_cuda_available():
        pytest.skip("requires NVIDIA CUDA")
    t = ActivityTable.load("mixed-two-vapor")
    mass, tk = t.recipe.states(np.full((3, 3), 0.5))
    mass[1, 2] = 1e-14
    mass[2, 1:] *= 1e-14
    values, errors, reasons = t.prepare(device=device).evaluate(mass, tk)
    truth, codes = NodeEvaluator(t.recipe, device).evaluate(mass, tk)
    assert np.all(codes >= 0)
    np.testing.assert_array_equal(reasons, [0, 3, 3])
    np.testing.assert_array_equal(errors[1:], 0.0)
    assert np.all(values[1:] > 0)
    np.testing.assert_allclose(
        np.log(values[1:]), np.log(truth[1:]), atol=1e-12, rtol=0
    )


def test_custom_recipe():
    r = load_recipe("mixed-water")
    t = build_table(
        replace(r, bounds=((2.0, 4.0), (292.0, 301.0)), max_level=6)
    )
    assert t.metadata["model"]["revision"]
    assert t.arrays["coefficients"].flags.writeable is False


def test_cuda_cross_build(tmp_path):
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("requires NVIDIA CUDA")
    for origin in ("cpu", "cuda:0"):
        t = build_table("mixed-water", device=origin)
        path = tmp_path / "table.npz"
        t.save(path)
        m, tk = t.recipe.states(
            np.random.default_rng(100).uniform(size=(129, 2))
        )
        tk[-1] = 325.0
        for dest in ("cpu", "cuda:0"):
            a, _, reason = (
                ActivityTable.load(path).prepare(device=dest).evaluate(m, tk)
            )
            truth, code = NodeEvaluator(t.recipe, dest).evaluate(m, tk)
            assert not np.any(code < 0)
            assert np.max(abs(a - truth)) <= 1e-4
            assert reason[-1] == 3 and (reason[:-1] == 0).all()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_coupled_trace_failure_retains_original_row(device, tmp_path):
    from particula_aiomfac import ConvergenceError
    from particula_aiomfac.sparse import TableRecipe

    recipe = TableRecipe(
        name="coupled",
        components=(
            ("water", {16: 1}),
            ("NH4HSO4", {204: 1, 248: 1}),
            ("KHCO3", {203: 1, 250: 1}),
        ),
        dry_weights=(0, 1, 1),
        targets=("water",),
        variable_dry=(),
        bounds=((3.0, 3.1), (295.0, 296.0)),
        log_tolerances=(4e-5,),
        max_level=1,
    )
    if device != "cpu" and not pytest.importorskip("warp").is_cuda_available():
        pytest.skip("requires NVIDIA CUDA")
    model = build_table(recipe, device=device).prepare(device=device)
    with pytest.raises(ConvergenceError) as exc:
        model.evaluate(
            [[0, 0, 0], [0.999999998, 1.0e-9, 1.0e-9]],
            [295.0, 295.0],
            active=[0, 1],
        )
    assert exc.value.batch_index == (1,)
    failed = build_table(
        replace(recipe, bounds=((19.9, 20.1), (294.9, 295.1))), device=device
    )
    artifact = tmp_path / "failed.npz"
    failed.save(artifact)
    saved = ActivityTable.load(artifact)
    assert np.isinf(saved.arrays["errors"]).all()
    failures = saved.metadata["failures"]
    assert failures and all(f["backend_status"] == -5 for f in failures)
    assert len({f["node_index"] for f in failures}) == len(failures)
    assert all(
        f["temperature_K"] > 0 and len(f["analytical_masses"]) == 3
        for f in failures
    )


def test_box_flux_guard_on_permuted_layout_and_digest_mismatch(tmp_path):
    from pathlib import Path
    from importlib.resources import files
    import runpy
    from particula_aiomfac.adapter import AIOMFACActivity

    pytest.importorskip("particula")
    root = Path(__file__).resolve().parents[3]
    example = (
        "docs/studies/examples/thermodynamics/activity/aiomfac_closed_box.py"
    )
    path = root / example
    if not path.is_file():
        path = files("particula").joinpath(example)
    lib = runpy.run_path(str(path))
    b = lib["make_box"](permuted=True, table="mixed-two-vapor")
    p, g = b.aerosol.particles, b.aerosol.gas
    exact = lib["make_box"](permuted=True).process
    # Midway between the approximate and exact water equilibrium pressures
    # would reverse the sign on slot zero without the flux guard.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        args = p, g, 298.0, np.maximum(p.radii[0], 1e-10), b.sources, b.targets
        approx = b.process._native_pressure_terms(*args)[1][0, b.targets[0]]
        truth = exact._native_pressure_terms(*args)[1][0, b.targets[0]]
        assert abs(approx - truth) > 1e-12
        gas_lane = b.sources[0]
        g.concentration[0, gas_lane] = (
            (approx + truth)
            * 0.5
            * g.molar_mass[gas_lane]
            / (lib["GAS_CONSTANT"] * 298.0)
        )
        expected = exact._native_pressure_terms(*args)[0]
        got = b.process._native_pressure_terms(*args)[0]
    np.testing.assert_allclose(got[0], expected[0], atol=1e-12)
    assert b.process.activity_strategy.counts[5] > 0
    assert b.process.activity_strategy.counts[2] > 0
    t = build_table("mixed-water")
    path = tmp_path / "portable.npz"
    t.save(path)
    p = AIOMFACActivity(
        t.recipe.mixture,
        dict(enumerate(t.recipe.mixture.components)),
        evaluation="hybrid",
        table=path,
    )
    frozen = p.prepared_identity()
    build_table("mixed-water", tolerance_scale=2).save(path)
    with pytest.raises(ValueError, match="incompatible"):
        AIOMFACActivity.from_identity(frozen)


def test_unqualified_cell_and_corrupt_arrays():
    t = ActivityTable.load("mixed-water")
    arrays = dict(t.arrays)
    err = arrays["errors"].copy()
    err[0, 0] = np.inf
    arrays["errors"] = err
    custom = ActivityTable(t.metadata, arrays)
    err[:] = 0
    with pytest.raises(ValueError):
        custom.arrays["coefficients"].setflags(write=True)
    m, tk = custom.recipe.states(np.array([[0.1, 0.1], [0.5, 0.5]]))
    _, _, why = custom.prepare().evaluate(m, tk)
    np.testing.assert_array_equal(why, [4, 0])
    arrays["levels"] = arrays["levels"][:, :1]
    with pytest.raises(ValueError, match="levels"):
        ActivityTable(t.metadata, arrays)


def test_custom_recipe_rejects_invalid_chemistry_and_bounds():
    r = load_recipe("mixed-water")
    with pytest.raises(ValueError, match="subgroup"):
        replace(r, components=(("water", {16.5: 1}), *r.components[1:]))
    with pytest.raises(ValueError, match="bounds"):
        replace(r, bounds=((0, 999), (292, 301)))
    with pytest.raises(ValueError, match="neutral"):
        build_table(replace(r, targets=("NaCl",)))
