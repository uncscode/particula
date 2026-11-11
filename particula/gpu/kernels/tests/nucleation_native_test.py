"""Direct/prepared native GPU source volume and species-map regressions."""

from dataclasses import replace

import numpy as np
import pytest

from particula.gpu.kernels.tests.nucleation_test import (
    _config,
    _exhaustion_buffers,
    _sidecars,
)
from particula.util.constants import AVOGADRO_NUMBER

pytestmark = [pytest.mark.warp, pytest.mark.gpu_parity]


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Require Warp CPU and conditionally exercise native CUDA."""
    wp = pytest.importorskip("warp")
    if request.param == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def setup(device, volume=0.25, capacity=20):
    """Create exact-arithmetic gas molecules and unequal carrier widths."""
    from particula.dynamics.nucleation.species_mapping import (
        NucleationSpeciesMap,
    )
    from particula.gas import GasData
    from particula.gpu import to_warp_gas_data, to_warp_particle_data
    from particula.gpu.kernels.nucleation import NucleationExhaustionControls
    from particula.gpu.nucleation_species_mapping import (
        prepare_nucleation_species_map,
    )
    from particula.particles import ParticleData

    masses = np.zeros((1, capacity, 3), dtype=np.float64)
    masses[0, 0] = [2.0, 3.0, 4.0]
    counts = np.zeros((1, capacity), dtype=np.float64)
    counts[0, 0] = 2.0
    cpu = ParticleData(
        masses=masses,
        concentration=counts,
        charge=np.zeros_like(counts),
        density=np.ones(3),
        volume=np.array([volume]),
        distribution_type="particle_resolved",
    )
    gas_cpu = GasData(
        name=["inert", "B", "A", "spectator"],
        molar_mass=np.full(4, AVOGADRO_NUMBER),
        concentration=np.array([[7.0, 100.0, 100.0, 9.0]]),
        partitioning=np.array([False, True, True, False]),
    )
    p = to_warp_particle_data(cpu, device=device)
    g = to_warp_gas_data(gas_cpu, device=device)
    binding = prepare_nucleation_species_map(
        p,
        g,
        gas_names=gas_cpu.name,
        species_map=NucleationSpeciesMap(tuple(gas_cpu.name), ((2, 0), (1, 1))),
    )
    scratch, finalized, diagnostics = _sidecars(1, capacity, 4, device)
    buffers = _exhaustion_buffers(1, capacity, 3, device)
    config = _config(
        coefficient=0.04, precursor_index=2, molecule_counts=(0, 2, 1, 0)
    )
    kwargs = dict(
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
        exhaustion_controls=NucleationExhaustionControls(False, False),
        exhaustion_buffers=buffers,
        temperature=300.0,
        species_binding=binding,
    )
    return p, g, config, kwargs


def density_inventory(p, g):
    """Compute independent mapped density, including particle-only mass."""
    result = np.sum(
        p.masses.numpy() * p.concentration.numpy()[..., None], axis=1
    )
    result /= p.volume.numpy()[:, None]
    result[:, 0] += g.concentration.numpy()[:, 2]
    result[:, 1] += g.concentration.numpy()[:, 1]
    return result


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_native_gpu_unit_count_and_independent_species_widths(device, volume):
    """One slot per event stores count one and consumes density in gas order."""
    from particula.gpu.kernels.nucleation import nucleation_step_gpu

    p, g, config, kwargs = setup(device, volume)
    before = density_inventory(p, g)
    assert nucleation_step_gpu(p, g, config, 1.0, **kwargs) == (p, g)
    number = int(4 * volume)
    np.testing.assert_array_equal(
        p.concentration.numpy()[0, 1 : number + 1], 1.0
    )
    np.testing.assert_array_equal(
        p.masses.numpy()[0, 1 : number + 1],
        np.tile([1.0, 2.0, 0.0], (number, 1)),
    )
    np.testing.assert_array_equal(
        g.concentration.numpy(), [[7.0, 92.0, 96.0, 9.0]]
    )
    np.testing.assert_allclose(
        density_inventory(p, g), before, rtol=1e-12, atol=1e-30
    )


def test_native_gpu_scaling_preserves_gas_and_existing_density(device):
    """Scaling reduces counts once; inert gas never receives a scaling factor."""
    import warp as wp

    from particula.gpu.kernels.nucleation import (
        NucleationExhaustionControls,
        nucleation_step_gpu,
    )

    p, g, config, kwargs = setup(device, volume=1.0, capacity=3)
    before = density_inventory(p, g)
    buffers = kwargs["exhaustion_buffers"]
    wp.copy(
        buffers.requested_scale,
        wp.array([0.5], dtype=wp.float64, device=device),
    )
    wp.copy(
        buffers.minimum_scale, wp.array([0.5], dtype=wp.float64, device=device)
    )
    wp.copy(
        buffers.minimum_volume,
        wp.array([0.01], dtype=wp.float64, device=device),
    )
    kwargs["exhaustion_controls"] = NucleationExhaustionControls(False, True)
    nucleation_step_gpu(p, g, config, 1.0, **kwargs)
    np.testing.assert_array_equal(p.volume.numpy(), [0.5])
    np.testing.assert_array_equal(p.concentration.numpy(), [[1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(
        g.concentration.numpy(), [[7.0, 92.0, 96.0, 9.0]]
    )
    np.testing.assert_allclose(
        density_inventory(p, g), before, rtol=1e-12, atol=1e-30
    )


def test_native_prepared_repeats_without_setup_and_reads_live_gas(
    device, monkeypatch
):
    """Retained arrays drive repeated enqueue with no host setup or readback."""
    import warp as wp

    from particula.gpu.kernels import nucleation as module

    p, g, config, kwargs = setup(device)
    prepared = module._prepare_nucleation_step_gpu(p, g, config, 1.0, **kwargs)
    module._enqueue_prepared_nucleation_call(prepared)
    module._observe_prepared_nucleation_call(prepared)
    # Restore precursor density explicitly so the next admitted event is integral.
    wp.copy(
        g.concentration,
        wp.array([[7.0, 92.0, 100.0, 9.0]], dtype=wp.float64, device=device),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("native prepared source performed host setup")

    with monkeypatch.context() as patch:
        for name in ("zeros", "empty", "array", "full", "synchronize"):
            patch.setattr(wp, name, forbidden)
        for name in (
            "p1_status",
            "p2_status",
            "p3_status",
            "p4_status",
            "p5_status",
        ):
            patch.setattr(getattr(prepared, name), "numpy", forbidden)
        module._enqueue_prepared_nucleation_call(prepared)
    module._observe_prepared_nucleation_call(prepared)
    np.testing.assert_array_equal(
        p.concentration.numpy()[0, :3], [2.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(
        g.concentration.numpy(), [[7.0, 84.0, 96.0, 9.0]]
    )


def test_native_binding_rejects_rebound_storage(device):
    """A binding cannot silently acquire replacement gas storage."""
    import warp as wp

    from particula.gpu.kernels.nucleation import _prepare_nucleation_step_gpu

    p, g, config, kwargs = setup(device)
    g.concentration = wp.zeros((1, 4), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="storage changed"):
        _prepare_nucleation_step_gpu(p, g, config, 1.0, **kwargs)


def test_native_gpu_resampling_then_source_preserves_density(device):
    """Resampling and unit-count source activation share the native population."""
    import warp as wp

    from particula.gpu.kernels.nucleation import (
        NucleationExhaustionControls,
        nucleation_step_gpu,
    )

    p, g, config, kwargs = setup(device, volume=0.5, capacity=4)
    wp.copy(
        p.masses,
        wp.array(
            [[[2.0, 3.0, 4.0]] * 3 + [[0.0, 0.0, 0.0]]],
            dtype=wp.float64,
            device=device,
        ),
    )
    wp.copy(
        p.concentration,
        wp.array([[2.0, 3.0, 5.0, 0.0]], dtype=wp.float64, device=device),
    )
    buffers = kwargs["exhaustion_buffers"]
    wp.copy(
        buffers.resampling_releasable_counts,
        wp.array([2], dtype=wp.int32, device=device),
    )
    kwargs["exhaustion_controls"] = NucleationExhaustionControls(True, False)
    before = density_inventory(p, g)
    nucleation_step_gpu(p, g, config, 1.0, **kwargs)
    np.testing.assert_array_equal(buffers.required_release_counts.numpy(), [1])
    np.testing.assert_array_equal(
        p.concentration.numpy(), [[5.0, 5.0, 1.0, 1.0]]
    )
    np.testing.assert_allclose(
        density_inventory(p, g), before, rtol=1e-12, atol=1e-30
    )


@pytest.mark.parametrize(
    "failure", ["density", "partitioning", "gas", "fractional"]
)
def test_native_prepared_invalid_live_payload_preserves_particles_and_gas(
    device, failure
):
    """Invalid live state gates the source before any primary writer."""
    import warp as wp

    from particula.gpu.kernels import nucleation as module

    p, g, config, kwargs = setup(device)
    prepared = module._prepare_nucleation_step_gpu(p, g, config, 1.0, **kwargs)
    if failure == "density":
        wp.copy(p.volume, wp.array([1e-309], dtype=wp.float64, device=device))
    elif failure == "partitioning":
        wp.copy(
            g.partitioning,
            wp.array([[0, 0, 1, 0]], dtype=wp.int32, device=device),
        )
    elif failure == "gas":
        wp.copy(
            g.concentration,
            wp.array(
                [[7.0, -1.0, 100.0, 9.0]], dtype=wp.float64, device=device
            ),
        )
    else:
        wp.copy(
            g.concentration,
            wp.array(
                [[7.0, 100.0, 50.0, 9.0]], dtype=wp.float64, device=device
            ),
        )
    arrays = [p.masses, p.concentration, p.charge, p.volume, g.concentration]
    before = [array.numpy().copy() for array in arrays]
    module._enqueue_prepared_nucleation_call(prepared)
    with pytest.raises(ValueError):
        module._observe_prepared_nucleation_call(prepared)
    for array, expected in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(array.numpy(), expected)


@pytest.mark.parametrize("code", [1, 2, 3])
def test_native_source_requires_supported_population_and_binding_before_payload(
    code,
):
    """A tag cannot infer species order or broaden the source algorithm."""
    from particula.gpu.kernels.nucleation import _prepare_nucleation_step_gpu

    class Tagged:
        distribution_kind = code

        def __getattr__(self, name):
            pytest.fail(f"invalid source admission accessed {name}")

    message = (
        "requires species_binding" if code == 3 else "only particle_resolved"
    )
    with pytest.raises(ValueError, match=message):
        _prepare_nucleation_step_gpu(
            Tagged(),
            None,
            None,
            None,
            scratch=None,
            finalized_demand=None,
            diagnostics=None,
            exhaustion_controls=None,
            exhaustion_buffers=None,
        )


def test_native_source_rejects_p4_alias_with_protected_vapor_pressure(device):
    """Every phase protects unused full-gas fields as well as active lanes."""
    from particula.gpu.kernels.nucleation import _prepare_nucleation_step_gpu

    p, g, config, kwargs = setup(device, capacity=4)
    buffers = kwargs["exhaustion_buffers"]
    kwargs["exhaustion_buffers"] = replace(
        buffers,
        resampling_buffers=replace(
            buffers.resampling_buffers, replacement_charge=g.vapor_pressure
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        _prepare_nucleation_step_gpu(p, g, config, 1.0, **kwargs)
