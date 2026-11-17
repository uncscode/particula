"""Selective physical forcing without a gas-array round trip."""

import numpy as np
import pytest

from particula.dynamics import Dilution, DilutionStrategy
from particula.execution.gpu_coupled import GPUCoupledSimulation
from particula.execution.tests.gpu_multimaterial_condensation_test import (
    material_case,
)
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    WaterBuckStrategy,
)
from particula.util.constants import GAS_CONSTANT

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forcing_retains_other_species_and_evaluates_humidity_on_device(
    device, monkeypatch
):
    """Ice/water forcing matches an independent CPU observation oracle."""
    if device != "cpu" and (
        not wp.is_cuda_available() or not wp.is_conditional_graph_supported()
    ):
        pytest.skip("CUDA unavailable")
    initial, strategy = material_case(3)
    initial.environment.temperature[:] = [265, 305]
    strategy.vapor_pressure_strategy = [
        ConstantVaporPressureStrategy(10)
    ] * 3 + [WaterBuckStrategy()]
    events = (Dilution(DilutionStrategy(0.0)),)
    with GPUCoupledSimulation(
        initial, events, strategy, device=device, time_step=1e-4
    ) as owner:
        gas = owner._events._session.gas.concentration
        original_numpy = wp.array.numpy

        def status_only(array, *args, **kwargs):
            assert array.size == 1, "forcing downloaded physical state"
            return original_numpy(array, *args, **kwargs)

        ratios = np.array([0.7, 1.2])
        with monkeypatch.context() as patch:
            patch.setattr(wp.array, "numpy", status_only)
            owner.update_gas_species(
                {"gas_1": 3e-4}, saturation_ratios={"gas_3": ratios}
            )
        assert owner._events._session.gas.concentration is gas
        actual = owner.snapshot().gas.concentration
        expected = (
            ratios
            * WaterBuckStrategy().pure_vapor_pressure(
                initial.environment.temperature
            )
            * initial.gas.molar_mass[3]
            / (GAS_CONSTANT * initial.environment.temperature)
        )
        np.testing.assert_allclose(actual[:, 3], expected, rtol=2e-14)
        np.testing.assert_array_equal(actual[:, 1], 3e-4)
        np.testing.assert_array_equal(
            actual[:, [0, 2]], initial.gas.concentration[:, [0, 2]]
        )
        before = actual.copy()
        for values, ratios in (
            ({"unknown": 0}, None),
            ({"gas_1": -1}, None),
            ({"gas_1": np.nan}, None),
            ({"gas_1": [1, 2, 3]}, None),
            ({"gas_1": 0}, {"gas_1": 1}),
            ({}, {}),
        ):
            with pytest.raises(ValueError):
                owner.update_gas_species(values, saturation_ratios=ratios)
            np.testing.assert_array_equal(
                owner.snapshot().gas.concentration, before
            )
        owner.step()
        assert owner.time == 1e-4
