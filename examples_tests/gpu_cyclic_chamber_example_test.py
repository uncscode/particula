"""Compact cyclic-chamber observations survive publication and saved-data plots."""

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scripts.docs_examples import import_study, write_plot_data
from scripts.gpu_studies.cases import StudyCase, build_case

example = import_study("gpu-prepared-cyclic-chamber")


def test_chamber_observations_save_and_replot_without_gpu(
    tmp_path, monkeypatch
):
    """Exercise the real observation and JSON writer before an expensive run."""
    initial, _, _ = build_case(
        StudyCase("plot-data", "full", 32, 8, occupancy=0.75, time_step=1.0)
    )
    diagnostics = SimpleNamespace(
        number_concentration=np.arange(32, dtype=float) + 1.0,
        temperature=np.linspace(278.0, 305.0, 32),
        saturation_ratio=np.full((32, 3), 1.2),
    )
    owner = SimpleNamespace(
        snapshot=lambda: initial, diagnostics=lambda: diagnostics
    )
    first, snapshot = example.frame(owner, 0)
    second, _ = example.frame(owner, 20)
    assert snapshot is initial
    path = tmp_path / "data.json"
    write_plot_data(path, {"frames": [first, second]})
    loaded = json.loads(path.read_text())
    np.testing.assert_array_equal(
        loaded["frames"][0]["temperature"], diagnostics.temperature
    )
    np.testing.assert_array_equal(
        loaded["frames"][0]["number"], diagnostics.number_concentration
    )
    diagnostics.temperature[:] = 0
    assert np.all(np.asarray(first["temperature"]) >= 278)

    def forbidden():
        raise AssertionError("Saved-data plotting must not request CUDA")

    monkeypatch.setattr(example.wp, "is_cuda_available", forbidden)
    try:
        example.plot_history(loaded["frames"])
        figure = plt.gcf()
        assert len(figure.axes) == 5
        figure.savefig(tmp_path / "figure.png")
        assert (tmp_path / "figure.png").stat().st_size > 0
    finally:
        plt.close("all")
