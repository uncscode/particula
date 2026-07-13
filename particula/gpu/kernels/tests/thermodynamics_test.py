"""Tests for the caller-owned GPU thermodynamic sidecar."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.warp


def _warp():
    """Import Warp at test runtime to preserve marker deselection."""
    return pytest.importorskip("warp")


def _config(n_species: int = 2):
    """Build a valid mixed constant/Buck CPU sidecar and gas fingerprint."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig

    masses = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    return ThermodynamicsConfig(
        modes=wp.array([0, 1][:n_species], dtype=wp.int32, device="cpu"),
        parameters=wp.zeros((n_species, 4), dtype=wp.float64, device="cpu"),
        molar_mass_reference=wp.array(masses, dtype=wp.float64, device="cpu"),
    ), wp.array(masses, dtype=wp.float64, device="cpu")


def _gas(
    n_boxes: int = 1,
    n_species: int = 2,
    sentinel: float = -123.0,
    device: str = "cpu",
):
    """Build a CPU Warp gas container with a nonphysical output sentinel."""
    wp = _warp()
    from particula.gpu import WarpGasData

    gas = WarpGasData()
    gas.molar_mass = wp.array(
        np.linspace(0.018, 0.05, n_species, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    gas.concentration = wp.zeros(
        (n_boxes, n_species), dtype=wp.float64, device=device
    )
    gas.vapor_pressure = wp.array(
        np.full((n_boxes, n_species), sentinel, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    gas.partitioning = wp.ones(n_species, dtype=wp.int32, device=device)
    return gas


def _refresh_config(gas, modes: np.ndarray, parameters: np.ndarray):
    """Create a valid refresh sidecar matching the supplied gas species."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig

    return ThermodynamicsConfig(
        modes=wp.array(modes, dtype=wp.int32, device=gas.molar_mass.device),
        parameters=wp.array(
            parameters, dtype=wp.float64, device=gas.molar_mass.device
        ),
        molar_mass_reference=wp.array(
            gas.molar_mass.numpy(),
            dtype=wp.float64,
            device=gas.molar_mass.device,
        ),
    )


@pytest.mark.gpu_parity
def test_refresh_constant_vapor_pressure_overwrites_all_cells() -> None:
    """Constant refresh overwrites every output cell in multiple boxes."""
    wp = _warp()
    from particula.gas.vapor_pressure_strategies import (
        ConstantVaporPressureStrategy,
    )
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu

    gas = _gas(n_boxes=3, n_species=1)
    pressure = 2345.6
    config = _refresh_config(
        gas,
        np.array([0], dtype=np.int32),
        np.array([[pressure, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    temperature = wp.array(
        [260.0, 273.15, 298.15], dtype=wp.float64, device="cpu"
    )

    refresh_vapor_pressure_gpu(config, gas, temperature)

    expected = ConstantVaporPressureStrategy(pressure).pure_vapor_pressure(
        temperature.numpy()
    )
    result = gas.vapor_pressure.numpy()
    assert result.dtype == np.float64
    assert result.shape == (3, 1)
    npt.assert_allclose(result[:, 0], expected, rtol=1e-12, atol=0.0)
    assert np.all(result != -123.0)


@pytest.mark.gpu_parity
@pytest.mark.parametrize("temperature_value", [260.0, 273.15, 298.15])
def test_refresh_buck_matches_cpu_reference(
    temperature_value: float,
) -> None:
    """Buck refresh matches the canonical CPU ice and water equations."""
    wp = _warp()
    from particula.gas.properties.vapor_pressure_module import (
        get_buck_vapor_pressure,
    )
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu

    gas = _gas(n_species=1)
    config = _refresh_config(
        gas,
        np.array([1], dtype=np.int32),
        np.array([[9.0, 8.0, 7.0, 6.0]], dtype=np.float64),
    )
    temperature = wp.array([temperature_value], dtype=wp.float64, device="cpu")

    refresh_vapor_pressure_gpu(config, gas, temperature)

    npt.assert_allclose(
        gas.vapor_pressure.numpy()[0, 0],
        get_buck_vapor_pressure(temperature_value),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.gpu_parity
def test_refresh_mixed_models_retains_species_order() -> None:
    """One launch refreshes mixed constant and Buck columns in order."""
    wp = _warp()
    from particula.gas.properties.vapor_pressure_module import (
        get_buck_vapor_pressure,
    )
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu

    gas = _gas(n_boxes=3)
    config = _refresh_config(
        gas,
        np.array([0, 1], dtype=np.int32),
        np.array([[101.0, 0.0, 0.0, 0.0], [4.0, 3.0, 2.0, 1.0]]),
    )
    temperatures = np.array([260.0, 273.15, 298.15], dtype=np.float64)
    refresh_vapor_pressure_gpu(
        config, gas, wp.array(temperatures, dtype=wp.float64, device="cpu")
    )

    expected = np.column_stack(
        (np.full(3, 101.0), get_buck_vapor_pressure(temperatures))
    )
    npt.assert_allclose(gas.vapor_pressure.numpy(), expected, rtol=1e-12)


@pytest.mark.gpu_parity
def test_refresh_buck_ignores_reserved_parameters() -> None:
    """Buck output is independent of its fixed-schema reserved slots."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu

    temperature = wp.array([260.0, 298.15], dtype=wp.float64, device="cpu")
    zero_gas = _gas(n_boxes=2, n_species=1)
    reserved_gas = _gas(n_boxes=2, n_species=1)
    zero_config = _refresh_config(
        zero_gas,
        np.array([1], dtype=np.int32),
        np.zeros((1, 4), dtype=np.float64),
    )
    reserved_config = _refresh_config(
        reserved_gas,
        np.array([1], dtype=np.int32),
        np.array([[9.0, 8.0, 7.0, 6.0]], dtype=np.float64),
    )

    refresh_vapor_pressure_gpu(zero_config, zero_gas, temperature)
    refresh_vapor_pressure_gpu(reserved_config, reserved_gas, temperature)

    npt.assert_allclose(
        reserved_gas.vapor_pressure.numpy(),
        zero_gas.vapor_pressure.numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_refresh_is_concrete_module_only_api() -> None:
    """Refresh is exported only from its intentionally concrete module."""
    import particula.gpu.kernels as kernels
    import particula.gpu.kernels.thermodynamics as thermodynamics

    assert "refresh_vapor_pressure_gpu" in thermodynamics.__all__
    assert "refresh_vapor_pressure_gpu" not in kernels.__all__
    assert not hasattr(kernels, "refresh_vapor_pressure_gpu")


@pytest.mark.parametrize(
    ("temperature", "message"),
    [
        (np.array([298.15], dtype=np.float64), "temperature"),
        (np.array([298.15], dtype=np.float32), "temperature"),
        (np.array([0.0], dtype=np.float64), "temperature"),
        (np.array([-1.0], dtype=np.float64), "temperature"),
        (np.array([np.nan], dtype=np.float64), "temperature"),
        (np.array([np.inf], dtype=np.float64), "temperature"),
        (np.array([298.15, 300.0], dtype=np.float64), "temperature"),
    ],
)
def test_refresh_invalid_temperature_does_not_mutate_output(
    temperature: np.ndarray,
    message: str,
) -> None:
    """Invalid temperature inputs fail before touching vapor-pressure output."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu

    gas = _gas(n_species=1)
    config = _refresh_config(
        gas,
        np.array([0], dtype=np.int32),
        np.array([[100.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    before = gas.vapor_pressure.numpy().copy()
    invalid = (
        temperature
        if temperature.shape == (1,) and temperature[0] == 298.15
        else wp.array(
            temperature,
            dtype=wp.float32 if temperature.dtype == np.float32 else wp.float64,
            device="cpu",
        )
    )

    with pytest.raises(ValueError, match=message):
        refresh_vapor_pressure_gpu(config, gas, invalid)

    assert np.array_equal(gas.vapor_pressure.numpy(), before)


def test_refresh_rejects_host_gas_field_without_mutation() -> None:
    """Host gas buffers fail validation before an output mutation is possible."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu

    gas = _gas(n_species=1)
    config = _refresh_config(
        gas,
        np.array([0], dtype=np.int32),
        np.array([[100.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    invalid_gas = SimpleNamespace(
        molar_mass=gas.molar_mass,
        vapor_pressure=np.array([[-123.0]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="gas.vapor_pressure"):
        refresh_vapor_pressure_gpu(
            config, invalid_gas, wp.array([298.15], dtype=wp.float64)
        )


def test_refresh_rejects_host_gas_molar_mass_without_mutation() -> None:
    """Host molar mass fails before the valid output buffer is overwritten."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu

    gas = _gas(n_species=1)
    config = _refresh_config(
        gas,
        np.array([0], dtype=np.int32),
        np.array([[100.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    invalid_gas = SimpleNamespace(
        molar_mass=np.array([0.018], dtype=np.float64),
        vapor_pressure=gas.vapor_pressure,
    )
    before = gas.vapor_pressure.numpy().copy()

    with pytest.raises(ValueError, match="gas.molar_mass"):
        refresh_vapor_pressure_gpu(
            config, invalid_gas, wp.array([298.15], dtype=wp.float64)
        )

    assert np.array_equal(gas.vapor_pressure.numpy(), before)


def test_refresh_rejects_invalid_sidecar_without_mutation() -> None:
    """Wrapper validates sidecars before it can overwrite vapor pressure."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import (
        ThermodynamicsConfig,
        refresh_vapor_pressure_gpu,
    )

    gas = _gas(n_species=1)
    valid = _refresh_config(
        gas,
        np.array([0], dtype=np.int32),
        np.array([[100.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    invalid = ThermodynamicsConfig(
        wp.array([99], dtype=wp.int32, device="cpu"),
        valid.parameters,
        valid.molar_mass_reference,
    )
    before = gas.vapor_pressure.numpy().copy()

    with pytest.raises(ValueError, match="thermodynamics.modes"):
        refresh_vapor_pressure_gpu(
            invalid, gas, wp.array([298.15], dtype=wp.float64, device="cpu")
        )

    assert np.array_equal(gas.vapor_pressure.numpy(), before)


@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_refresh_buck_matches_cpu_reference_on_cuda_when_available() -> None:
    """CUDA Buck refresh matches the explicit CPU reference when available."""
    wp = _warp()
    from particula.gas.properties.vapor_pressure_module import (
        get_buck_vapor_pressure,
    )
    from particula.gpu.kernels.thermodynamics import refresh_vapor_pressure_gpu
    from particula.gpu.tests.cuda_availability import cuda_available

    if not cuda_available(wp):
        pytest.skip("CUDA is unavailable")
    gas = _gas(n_boxes=2, n_species=1, device="cuda")
    config = _refresh_config(
        gas,
        np.array([1], dtype=np.int32),
        np.zeros((1, 4), dtype=np.float64),
    )
    temperatures = np.array([260.0, 298.15], dtype=np.float64)

    refresh_vapor_pressure_gpu(
        config, gas, wp.array(temperatures, dtype=wp.float64, device="cuda")
    )

    npt.assert_allclose(
        gas.vapor_pressure.numpy()[:, 0],
        get_buck_vapor_pressure(temperatures),
        rtol=1e-12,
        atol=0.0,
    )


def test_valid_config_preserves_identity_and_buffers() -> None:
    """A valid sidecar returns identically without changing caller buffers."""
    from particula.gpu.kernels.thermodynamics import (
        validate_thermodynamics_config,
    )

    config, gas_masses = _config()
    before = tuple(
        field.numpy().copy()
        for field in (
            config.modes,
            config.parameters,
            config.molar_mass_reference,
        )
    )

    result = validate_thermodynamics_config(
        config, 2, config.modes.device, gas_masses, "test"
    )

    assert result is config
    for expected, field in zip(
        before,
        (config.modes, config.parameters, config.molar_mass_reference),
        strict=True,
    ):
        assert np.array_equal(expected, field.numpy())


@pytest.mark.parametrize("value", [None, object()])
def test_non_config_is_rejected(value: object | None) -> None:
    """Missing and non-config sidecars report the thermodynamics field."""
    from particula.gpu.kernels.thermodynamics import (
        validate_thermodynamics_config,
    )

    config, gas_masses = _config()
    with pytest.raises(ValueError, match="thermodynamics"):
        validate_thermodynamics_config(
            value, 2, config.modes.device, gas_masses, "test"
        )


def test_invalid_modes_are_rejected() -> None:
    """Unsupported mode codes fail after structural validation."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import (
        ThermodynamicsConfig,
        validate_thermodynamics_config,
    )

    config, gas_masses = _config()
    invalid = ThermodynamicsConfig(
        modes=wp.array([0, -1], dtype=wp.int32, device="cpu"),
        parameters=config.parameters,
        molar_mass_reference=config.molar_mass_reference,
    )
    with pytest.raises(ValueError, match="thermodynamics.modes"):
        validate_thermodynamics_config(
            invalid, 2, invalid.modes.device, gas_masses, "test"
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -1.0])
def test_invalid_parameters_are_rejected(invalid_value: float) -> None:
    """Non-finite and negative parameter values are rejected."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import (
        ThermodynamicsConfig,
        validate_thermodynamics_config,
    )

    config, gas_masses = _config()
    parameters = np.zeros((2, 4), dtype=np.float64)
    parameters[0, 0] = invalid_value
    invalid = ThermodynamicsConfig(
        config.modes,
        wp.array(parameters, dtype=wp.float64, device="cpu"),
        config.molar_mass_reference,
    )
    with pytest.raises(ValueError, match="thermodynamics.parameters"):
        validate_thermodynamics_config(
            invalid, 2, invalid.modes.device, gas_masses, "test"
        )


def test_ordered_molar_mass_mismatch_is_rejected() -> None:
    """A permuted fingerprint is rejected even when values are otherwise equal."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import (
        ThermodynamicsConfig,
        validate_thermodynamics_config,
    )

    config, gas_masses = _config()
    invalid = ThermodynamicsConfig(
        config.modes,
        config.parameters,
        wp.array([0.05, 0.018], dtype=wp.float64, device="cpu"),
    )
    with pytest.raises(ValueError, match="molar_mass_reference"):
        validate_thermodynamics_config(
            invalid, 2, invalid.modes.device, gas_masses, "test"
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -1.0])
def test_invalid_molar_mass_references_are_rejected(
    invalid_value: float,
) -> None:
    """Non-finite and negative reference values are rejected."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import (
        ThermodynamicsConfig,
        validate_thermodynamics_config,
    )

    config, gas_masses = _config()
    references = config.molar_mass_reference.numpy()
    references[0] = invalid_value
    invalid = ThermodynamicsConfig(
        config.modes,
        config.parameters,
        wp.array(references, dtype=wp.float64, device="cpu"),
    )

    with pytest.raises(ValueError, match="molar_mass_reference"):
        validate_thermodynamics_config(
            invalid, 2, invalid.modes.device, gas_masses, "test"
        )


def test_config_is_frozen() -> None:
    """The sidecar object prevents accidental attribute replacement."""
    config, _ = _config()
    with pytest.raises(FrozenInstanceError):
        config.modes = None


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("modes", np.zeros(2, dtype=np.float64), "thermodynamics.modes"),
        ("modes", np.zeros((2, 1), dtype=np.int32), "thermodynamics.modes"),
        (
            "parameters",
            np.zeros((2, 3), dtype=np.float64),
            "thermodynamics.parameters",
        ),
        (
            "molar_mass_reference",
            np.zeros(2, dtype=np.float32),
            "thermodynamics.molar_mass_reference",
        ),
    ],
)
def test_invalid_metadata_is_rejected_without_readback(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    values: np.ndarray,
    message: str,
) -> None:
    """Structural sidecar errors fail before any device-buffer readback."""
    wp = _warp()
    import particula.gpu.kernels.thermodynamics as thermodynamics_module

    config, gas_masses = _config()
    warp_dtype = {
        np.dtype(np.float32): wp.float32,
        np.dtype(np.float64): wp.float64,
        np.dtype(np.int32): wp.int32,
    }[values.dtype]
    invalid = replace(
        config,
        **{field: wp.array(values, dtype=warp_dtype, device="cpu")},
    )
    readbacks: list[object] = []

    def _unexpected_readback(values: object) -> np.ndarray:
        readbacks.append(values)
        raise AssertionError("structural validation must not read buffers")

    monkeypatch.setattr(
        thermodynamics_module,
        "_read_array",
        _unexpected_readback,
    )

    with pytest.raises(ValueError, match=message):
        thermodynamics_module.validate_thermodynamics_config(
            invalid, 2, config.modes.device, gas_masses, "test"
        )

    assert readbacks == []


def test_valid_config_reads_each_required_buffer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid validation reads each sidecar and gas buffer no more than once."""
    import particula.gpu.kernels.thermodynamics as thermodynamics_module

    config, gas_masses = _config()
    original_read = thermodynamics_module._read_array
    readbacks: list[object] = []

    def _count_readbacks(values: object) -> np.ndarray:
        readbacks.append(values)
        return original_read(values)

    monkeypatch.setattr(
        thermodynamics_module,
        "_read_array",
        _count_readbacks,
    )

    result = thermodynamics_module.validate_thermodynamics_config(
        config, 2, config.modes.device, gas_masses, "test"
    )

    assert result is config
    assert readbacks == [
        config.modes,
        config.parameters,
        config.molar_mass_reference,
        gas_masses,
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("modes", np.zeros(2, dtype=np.int32)),
        ("parameters", np.zeros((2, 4), dtype=np.float64)),
        ("molar_mass_reference", np.zeros(2, dtype=np.float64)),
    ],
)
def test_host_array_field_is_rejected_without_readback(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: np.ndarray,
) -> None:
    """Every required field rejects host arrays before device data is read."""
    import particula.gpu.kernels.thermodynamics as thermodynamics_module

    config, gas_masses = _config()
    invalid = replace(config, **{field: replacement})
    monkeypatch.setattr(
        thermodynamics_module,
        "_read_array",
        lambda _: pytest.fail("structural validation must not read buffers"),
    )

    with pytest.raises(ValueError, match=f"thermodynamics.{field}"):
        thermodynamics_module.validate_thermodynamics_config(
            invalid, 2, config.modes.device, gas_masses, "test"
        )


def test_externally_mutated_config_is_not_cached() -> None:
    """A mutable device buffer is revalidated on every validator call."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import (
        validate_thermodynamics_config,
    )

    config, gas_masses = _config()
    validate_thermodynamics_config(
        config, 2, config.modes.device, gas_masses, "test"
    )
    wp.copy(
        config.parameters,
        wp.array(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=wp.float64,
            device="cpu",
        ),
    )

    with pytest.raises(ValueError, match="thermodynamics.parameters"):
        validate_thermodynamics_config(
            config, 2, config.modes.device, gas_masses, "test"
        )
