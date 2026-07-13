"""Tests for the caller-owned GPU thermodynamic sidecar."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
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
