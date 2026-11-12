"""Population provenance round trips and explicit species-map preparation."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import GasData
from particula.gpu import conversion
from particula.gpu.population_metadata import (
    particle_distribution_code,
    particle_distribution_type,
)
from particula.gpu.species_mapping import prepare_condensation_species_map
from particula.particles import ParticleData


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Exercise Warp CPU and optional native CUDA without substituting backends."""
    wp = pytest.importorskip("warp")
    from particula.gpu.tests.cuda_availability import cuda_available

    if request.param == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA unavailable")
    return request.param


def _particles(kind="particle_resolved", volume=0.25):
    """Build facade-free unequal-count, two-material, two-box physical state."""
    return ParticleData(
        masses=np.array(
            [[[2, 3], [5, 7]], [[11, 13], [17, 19]]], dtype=np.float64
        )
        * 1e-18,
        concentration=np.array([[2.0, 0.5], [3.0, 4.0]]),
        charge=np.array([[1.0, -1.0], [2.0, 0.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([volume, volume]),
        distribution_type=kind,
    )


def _mapped_state(device):
    """Use unequal particle/gas widths and a nonidentity complete gas order."""
    particles = conversion.to_warp_particle_data(_particles(), device=device)
    gas = conversion.to_warp_gas_data(
        GasData(
            name=["inert", "B", "A"],
            molar_mass=np.array([0.04, 0.03, 0.02]),
            concentration=np.array([[0.0, 2e-9, 3e-9], [1e-9, 4e-9, 5e-9]]),
            partitioning=np.array([False, True, True]),
        ),
        device=device,
    )
    species_map = CondensationSpeciesMap(("inert", "B", "A"), ((2, 0), (1, 1)))
    return particles, gas, species_map


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("copy", [True, False])
def test_resolved_raw_round_trip_and_independent_inventory(
    device, volume, copy
):
    """Tag survives while raw counts and per-particle masses remain unchanged."""
    source = _particles(volume=volume)
    before = {
        name: getattr(source, name).copy()
        for name in ("masses", "concentration", "charge", "density", "volume")
    }
    gpu = conversion.to_warp_particle_data(source, device=device, copy=copy)
    assert particle_distribution_code(gpu) == 3
    restored = conversion.from_warp_particle_data(gpu)
    assert restored.distribution_type == "particle_resolved"
    assert restored.radius_grid is None
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(source, name), expected)
        np.testing.assert_array_equal(getattr(restored, name), expected)
    # Independent species sums: 2*[2,3]+.5*[5,7], 3*[11,13]+4*[17,19].
    inventory = np.array([[6.5, 9.5], [101.0, 115.0]]) * 1e-18
    np.testing.assert_allclose(
        restored.species_mass_inventory, inventory, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        restored.species_mass_density,
        inventory / volume,
        rtol=1e-12,
        atol=1e-30,
    )


@pytest.mark.parametrize("kind,code", [("discrete", 1)])
def test_pmf_round_trip_preserves_meaning(device, kind, code):
    """Transfer preserves the explicit PMF density interpretation."""
    source = _particles(kind, 1.0)
    gpu = conversion.to_warp_particle_data(source, device=device)
    assert particle_distribution_code(gpu) == code
    restored = conversion.from_warp_particle_data(gpu, sync=True)
    assert restored.distribution_type == kind
    np.testing.assert_array_equal(restored.concentration, source.concentration)


@pytest.mark.parametrize("code", [-1, 0, 2, 4, 3.5, True, None, "3"])
def test_restore_rejects_bad_or_unsupported_tag_before_probe(monkeypatch, code):
    """No synchronization/readback occurs when provenance cannot be restored."""

    def forbidden():
        raise AssertionError("Warp must not be probed")

    monkeypatch.setattr(conversion, "_ensure_warp_available", forbidden)
    with pytest.raises(ValueError):
        conversion.from_warp_particle_data(
            SimpleNamespace(distribution_kind=code)
        )


def test_default_warp_tag_cannot_be_interpreted(device):
    """A fresh struct has an unset tag and cannot imply a population."""
    from particula.gpu.warp_types import WarpParticleData

    with pytest.raises(ValueError, match="explicit population metadata"):
        particle_distribution_type(WarpParticleData())


@pytest.mark.parametrize(
    "skip,forward,reverse",
    [
        ((), [2, 1], [-1, 1, 0]),
        ((2,), [-1, 1], [-1, 1, -1]),
        ((1, 2), [-1, -1], [-1, -1, -1]),
    ],
)
def test_explicit_binding_preserves_full_orders(device, skip, forward, reverse):
    """Map uploads preserve independent widths and never compact gas storage."""
    particles, gas, species_map = _mapped_state(device)
    old = gas.concentration.numpy().copy()
    names = ["inert", "B", "A"]
    binding = prepare_condensation_species_map(
        particles,
        gas,
        gas_names=names,
        species_map=species_map,
        skip_partitioning_indices=skip,
    )
    names.reverse()
    assert binding.gas_names == ("inert", "B", "A")
    assert binding.species_map is species_map
    assert binding.particles is particles and binding.gas is gas
    assert binding.device == particles.masses.device
    np.testing.assert_array_equal(binding.particle_to_gas.numpy(), forward)
    np.testing.assert_array_equal(binding.gas_to_particle.numpy(), reverse)
    np.testing.assert_array_equal(gas.concentration.numpy(), old)
    binding.validate(particles, gas)
    with pytest.raises(FrozenInstanceError):
        binding.gas_names = ("A", "B", "inert")


@pytest.mark.parametrize(
    "fault",
    [
        "names",
        "short_names",
        "target",
        "bool_skip",
        "outside_skip",
        "untagged",
        "pdf",
        "unknown",
        "dtype",
        "shape",
    ],
)
def test_binding_rejects_before_configuration_upload(
    device, monkeypatch, fault
):
    """Invalid provenance/schema fails without allocations or payload writes."""
    import warp as wp

    particles, gas, mapping = _mapped_state(device)
    names, skips = ("inert", "B", "A"), ()
    if fault == "names":
        names = ("A", "B", "inert")
    elif fault == "short_names":
        names = ("B", "A")
        mapping = CondensationSpeciesMap(names, ((0, 0), (1, 1)))
    elif fault == "target":
        mapping = CondensationSpeciesMap(names, ((2, 2),))
    elif fault == "bool_skip":
        skips = (True,)
    elif fault == "outside_skip":
        skips = (3,)
    elif fault in ("untagged", "pdf", "unknown"):
        particles.distribution_kind = {"untagged": 0, "pdf": 2, "unknown": 9}[
            fault
        ]
    elif fault == "dtype":
        gas = SimpleNamespace(
            **{
                name: getattr(gas, name)
                for name in (
                    "concentration",
                    "molar_mass",
                    "partitioning",
                    "vapor_pressure",
                )
            }
        )
        gas.molar_mass = wp.ones(3, dtype=wp.float32, device=device)
    elif fault == "shape":
        gas.concentration = wp.zeros((1, 3), dtype=wp.float64, device=device)
    before = particles.masses.numpy().copy(), gas.concentration.numpy().copy()

    # Observe allocation while retaining wp.array's type identity.
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid configuration must not upload")

    monkeypatch.setattr(wp.array, "__init__", forbidden)
    with pytest.raises(ValueError):
        prepare_condensation_species_map(
            particles,
            gas,
            gas_names=names,
            species_map=mapping,
            skip_partitioning_indices=skips,
        )
    monkeypatch.undo()
    np.testing.assert_array_equal(particles.masses.numpy(), before[0])
    np.testing.assert_array_equal(gas.concentration.numpy(), before[1])


@pytest.mark.parametrize("fault", ["carrier", "storage", "metadata"])
def test_binding_rejects_stale_reuse(device, fault):
    """Equal shape does not authorize a binding for different storage."""
    import warp as wp

    particles, gas, mapping = _mapped_state(device)
    binding = prepare_condensation_species_map(
        particles, gas, gas_names=mapping.gas_names, species_map=mapping
    )
    if fault == "carrier":
        particles = conversion.to_warp_particle_data(
            _particles(), device=device
        )
    elif fault == "storage":
        gas.concentration = wp.clone(gas.concentration)
    else:
        particles.distribution_kind = 1
    with pytest.raises(ValueError, match="binding"):
        binding.validate(particles, gas)


def test_binding_validation_has_no_device_work(device, monkeypatch):
    """Admission of a prepared map only inspects pinned host metadata."""
    import warp as wp

    particles, gas, mapping = _mapped_state(device)
    binding = prepare_condensation_species_map(
        particles, gas, gas_names=mapping.gas_names, species_map=mapping
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("binding validation must not perform device work")

    monkeypatch.setattr(wp.array, "__init__", forbidden)
    monkeypatch.setattr(wp.array, "numpy", forbidden)
    monkeypatch.setattr(wp, "launch", forbidden)
    monkeypatch.setattr(wp, "synchronize", forbidden)
    binding.validate(particles, gas)


@pytest.mark.parametrize("copy", [True, False])
def test_untagged_upload_rejects_before_backend_probe(monkeypatch, copy):
    """Raw CPU scratch is not a transferable population without a tag."""
    source = _particles(None)
    before = source.masses.copy(), source.concentration.copy()

    def forbidden():
        pytest.fail("untagged upload reached backend setup")

    monkeypatch.setattr(conversion, "_ensure_warp_available", forbidden)
    with pytest.raises(ValueError, match="explicit population metadata"):
        conversion.to_warp_particle_data(source, device="cpu", copy=copy)
    np.testing.assert_array_equal(source.masses, before[0])
    np.testing.assert_array_equal(source.concentration, before[1])


def test_missing_restore_tag_rejects_before_backend_probe(monkeypatch):
    """A missing tag cannot trigger synchronization or payload access."""

    def forbidden():
        pytest.fail("untagged restore reached backend setup")

    monkeypatch.setattr(conversion, "_ensure_warp_available", forbidden)
    with pytest.raises(ValueError, match="explicit population metadata"):
        conversion.from_warp_particle_data(SimpleNamespace())
