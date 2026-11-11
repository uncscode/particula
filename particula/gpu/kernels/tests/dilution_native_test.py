"""Independent native Warp dilution expectations and protected storage."""

from types import SimpleNamespace

import numpy as np
import pytest

from particula.dynamics.tests.native_runnable_test import state

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Warp CPU baseline; native CUDA is optional without a fallback."""
    if request.param == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def containers(kind, volume, device):
    """Transfer raw counts/densities explicitly with unequal species widths."""
    from particula.gpu import to_warp_gas_data, to_warp_particle_data

    aerosol = state(volume, kind)
    return (
        to_warp_particle_data(aerosol.particles, device=device),
        to_warp_gas_data(aerosol.gas, device=device),
    )


@pytest.mark.parametrize(
    "kind,volume",
    [
        ("discrete", 1.0),
        ("particle_resolved", 0.25),
        ("particle_resolved", 1.0),
        ("particle_resolved", 4.0),
    ],
)
@pytest.mark.parametrize("prepared", [False, True])
def test_native_decay_preserves_raw_basis_and_every_protected_field(
    device, kind, volume, prepared, monkeypatch
):
    """Independent exponential expectations hold through direct and pinned calls."""
    from particula.gpu.kernels import dilution as module

    particles, gas = containers(kind, volume, device)
    targets = (particles.concentration, gas.concentration)
    before = [a.numpy().copy() for a in targets]
    protected = (
        particles.masses,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.vapor_pressure,
        gas.partitioning,
    )
    protected_before = [a.numpy().copy() for a in protected]
    if prepared:
        call = module._prepare_dilution_step_gpu(particles, gas, 0.25, 2.0)

        def forbidden(*args, **kwargs):
            pytest.fail("prepared enqueue performed setup or synchronization")

        with monkeypatch.context() as patch:
            patch.setattr(module, "_prepare_dilution_step_gpu", forbidden)
            patch.setattr(wp, "synchronize", forbidden)
            patch.setattr(wp, "zeros", forbidden)
            patch.setattr(wp, "empty", forbidden)
            result = module._enqueue_prepared_dilution_call(call)
            module._enqueue_prepared_dilution_call(call)
    else:
        result = module.dilution_step_gpu(particles, gas, 0.25, 4.0)
    assert result[0] is particles and result[1] is gas
    wp.synchronize()
    for target, values in zip(targets, before, strict=True):
        np.testing.assert_allclose(
            target.numpy(), values * np.exp(-1), rtol=1e-12, atol=0
        )
    for array, values in zip(protected, protected_before, strict=True):
        np.testing.assert_array_equal(array.numpy(), values)


@pytest.mark.parametrize(
    "failure",
    [
        "volume_zero",
        "volume_nan",
        "pmf_volume",
        "alias",
        "counts_overflow",
        "mass_nan",
        "scratch_alias",
    ],
)
@pytest.mark.parametrize("time", [0.0, 1.0])
def test_native_preflight_rejects_without_primary_writes(device, failure, time):
    """Bad semantics and protected aliases cannot reach the dilution writers."""
    from particula.gpu.kernels.dilution import _prepare_dilution_step_gpu

    particles, gas = containers(
        "discrete" if failure == "pmf_volume" else "particle_resolved",
        1.0,
        device,
    )
    options = {}
    if failure.startswith("volume") or failure == "pmf_volume":
        value = {"volume_zero": 0.0, "volume_nan": np.nan, "pmf_volume": 4.0}[
            failure
        ]
        particles.volume.assign(np.array([value]))
    elif failure == "counts_overflow":
        particles.volume.assign(np.array([0.25]))
        particles.concentration.assign(np.array([[1e308, 2.0]]))
    elif failure == "mass_nan":
        particles.masses.assign(np.full((1, 2, 2), np.nan))
    elif failure == "scratch_alias":
        options["factors"] = particles.volume
    else:
        particles.charge = particles.concentration
    arrays = (
        particles.concentration,
        gas.concentration,
        particles.masses,
        particles.charge,
        particles.volume,
    )
    before = [array.numpy().copy() for array in arrays]
    with pytest.raises(ValueError):
        _prepare_dilution_step_gpu(particles, gas, 0.25, time, **options)
    for array, values in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(array.numpy(), values)


def test_pdf_rejects_before_payload_access():
    """Dilution does not implicitly enable the unsupported GPU PDF carrier."""
    from particula.gpu.kernels.dilution import _prepare_dilution_step_gpu

    with pytest.raises(ValueError, match="continuous_pdf"):
        _prepare_dilution_step_gpu(
            SimpleNamespace(distribution_kind=2), None, 0.0, 0.0
        )
