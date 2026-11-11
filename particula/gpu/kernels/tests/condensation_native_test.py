"""Independent native mapped, density-normalized direct/prepared science."""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import GasData
from particula.particles import ParticleData
from particula.util import constants

pytestmark = pytest.mark.warp


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Require installed Warp CPU and explicitly skip unavailable CUDA."""
    wp = pytest.importorskip("warp")
    from particula.gpu.tests.cuda_availability import cuda_available

    if request.param == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA unavailable")
    return request.param


def _state(
    device,
    kind="particle_resolved",
    volume=0.25,
    scenario="mixed",
    latent=True,
    single=False,
):
    """Facade-free state with unequal counts and independent material/gas widths."""
    import warp as wp

    from particula.gpu.conversion import to_warp_gas_data, to_warp_particle_data
    from particula.gpu.kernels.condensation import (
        CondensationActivitySurfaceConfig,
        CondensationScratchBuffers,
    )
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
    from particula.gpu.species_mapping import prepare_condensation_species_map

    masses = (
        np.array(
            [
                [[2, 3, 1], [5, 7, 2], [1, 2, 4]],
                [[4, 2, 1], [6, 8, 3], [2, 3, 4]],
            ],
            dtype=np.float64,
        )
        * 1e-18
    )
    if single:
        masses = masses[:, :, :1].copy()
    materials = masses.shape[2]
    particle = ParticleData(
        masses=masses,
        concentration=np.array([[2e9, 5e8, 0], [3e8, 4e9, 0]]),
        charge=np.array([[1.0, -1.0, 0.0], [2.0, 0.0, 0.0]]),
        density=np.array([1000.0, 1200.0, 1500.0])[:materials],
        volume=np.full(2, volume),
        distribution_type=kind,
    )
    gas = GasData(
        name=["inert", "B", "A", "unused"],
        molar_mass=np.array([0.04, 0.03, 0.02, 0.06]),
        concentration=np.array(
            [[0.0, 2e-9, 3e-9, 9e-9], [1e-9, 4e-9, 5e-9, 7e-9]]
        ),
        partitioning=np.array([False, True, True, True]),
    )
    vp = np.array([7.0, 2e-4, 1e-4, 11.0])
    if scenario == "uptake":
        vp[1:3] = 1e-120
    elif scenario == "evaporation":
        vp[1:3] = 100.0
    flags = np.tile(gas.partitioning, (2, 1))
    if scenario == "disabled":
        flags[0, 1] = False
        flags[1, 2] = False
    skips = (1,) if scenario == "skipped" else ()
    pairs = ((2, 0),) if single else ((2, 0), (1, 1))
    gp = to_warp_particle_data(particle, device=device)
    gg = to_warp_gas_data(gas, device=device)
    gg.partitioning.assign(flags.astype(np.int32))
    binding = prepare_condensation_species_map(
        gp,
        gg,
        gas_names=gas.name,
        species_map=CondensationSpeciesMap(tuple(gas.name), pairs),
        skip_partitioning_indices=skips,
    )

    def array(values, dtype=wp.float64):
        return wp.array(values, dtype=dtype, device=device)

    thermodynamics = ThermodynamicsConfig(
        modes=array(np.zeros(4, dtype=np.int32), wp.int32),
        parameters=array(np.column_stack((vp, np.zeros((4, 3))))),
        molar_mass_reference=array(gas.molar_mass),
    )
    refs = np.array([0.02, 0.03, 0.1])[:materials]
    kappas = np.array([0.0, 0.4, 0.2])[:materials]
    activity = CondensationActivitySurfaceConfig(
        activity_mode=0,
        surface_tension_mode=1,
        water_species_index=0,
        kappas=array(kappas),
        molar_mass_reference=array(refs),
    )
    surface = np.array([0.072, 0.04, 0.06])[:materials]
    accom = np.array([0.2, 0.7, 1.0, 0.3])
    diffusion = np.array([1e-5, 2e-5, 3e-5, 4e-5])
    heat = np.array([1e8, 2e6, 1e6, 3e8]) if latent else np.zeros(4)
    energy = wp.full((2, 4), np.nan, dtype=wp.float64, device=device)
    total = wp.full(masses.shape, np.nan, dtype=wp.float64, device=device)
    work = wp.full(masses.shape, np.nan, dtype=wp.float64, device=device)
    scratch = CondensationScratchBuffers(
        work_mass_transfer=work,
        total_mass_transfer=total,
        positive_mass_transfer_demand=wp.zeros(
            (2, 4), dtype=wp.float64, device=device
        ),
        negative_mass_transfer_release=wp.zeros(
            (2, 4), dtype=wp.float64, device=device
        ),
        positive_mass_transfer_scale=wp.zeros(
            (2, 4), dtype=wp.float64, device=device
        ),
    )
    kwargs = dict(
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
        thermodynamics=thermodynamics,
        species_binding=binding,
        activity_surface=activity,
        surface_tension=array(surface),
        mass_accommodation=array(accom),
        diffusion_coefficient_vapor=array(diffusion),
        scratch_buffers=scratch,
    )
    if latent:
        kwargs.update(latent_heat=array(heat), energy_transfer=energy)
    return SimpleNamespace(
        cpu_p=particle,
        cpu_g=gas,
        p=gp,
        g=gg,
        kwargs=kwargs,
        total=total,
        work=work,
        energy=energy,
        heat=heat,
        pairs=pairs,
        skips=skips,
        flags=flags,
        vp=vp,
        surface=surface,
        accom=accom,
        diffusion=diffusion,
        refs=refs,
        kappas=kappas,
    )


def _oracle(state):  # noqa: C901
    """Evaluate scalar equations and four conservative steps without helpers."""
    p, gas = state.cpu_p, state.cpu_g
    masses, gases = p.masses.copy(), gas.concentration.copy()
    weights = p.concentration.copy()
    if p.distribution_type == "particle_resolved":
        weights /= p.volume[:, None]
    total = np.zeros_like(masses)
    energy = np.zeros_like(gases)
    temp, pressure, dt = (
        state.kwargs[x] for x in ("temperature", "pressure", "time_step")
    )
    rgas = constants.GAS_CONSTANT
    mu = (
        constants.REF_VISCOSITY_AIR_STP
        * (temp / constants.REF_TEMPERATURE_STP) ** 1.5
        * (constants.REF_TEMPERATURE_STP + constants.SUTHERLAND_CONSTANT)
        / (temp + constants.SUTHERLAND_CONSTANT)
    )
    mean_path = (
        2
        * mu
        / pressure
        / np.sqrt(8 * constants.MOLECULAR_WEIGHT_AIR / (np.pi * rgas * temp))
    )
    for _ in range(4):
        proposal = np.zeros_like(masses)
        for b in range(masses.shape[0]):
            for i in range(masses.shape[1]):
                if weights[b, i] == 0:
                    continue
                vols = masses[b, i] / p.density
                particle_volume = sum(vols)
                if particle_volume == 0:
                    continue
                radius = (3 * particle_volume / (4 * np.pi)) ** (1 / 3)
                density = sum(masses[b, i]) / particle_volume
                config = state.kwargs.get("activity_surface")
                tension = np.dot(state.surface, vols) / particle_volume
                kn = mean_path / radius
                moles = masses[b, i] / state.refs
                activity = moles[0] / sum(moles) if sum(moles) else 0.0
                for g, material in state.pairs:
                    if g in state.skips or not state.flags[b, g]:
                        continue
                    alpha = state.accom[g]
                    transition = (
                        0.75
                        * alpha
                        * (1 + kn)
                        / (kn * kn + kn + 0.283 * alpha * kn + 0.75 * alpha)
                    )
                    transport = (
                        4 * np.pi * radius * state.diffusion[g] * transition
                    )
                    molar = gas.molar_mass[g]
                    if config is None or config.surface_tension_mode == 0:
                        tension = state.surface[material]
                    surface_pressure = state.vp[g] * np.exp(
                        2 * tension * molar / (rgas * temp * density * radius)
                    )
                    if material == 0 and config is not None:
                        if config.activity_mode == 1:
                            activity = (
                                vols[0]
                                / (vols[0] + np.dot(state.kappas[1:], vols[1:]))
                                if vols[0]
                                else 0.0
                            )
                        surface_pressure *= activity
                    pressure_delta = (
                        gases[b, g] * rgas * temp / molar - surface_pressure
                    )
                    rate = transport * pressure_delta * molar / (rgas * temp)
                    heat = state.heat[g]
                    if heat:
                        r_specific = rgas / molar
                        conductivity = 1e-3 * (4.39 + 0.071 * temp)
                        correction = 1 + (
                            state.diffusion[g]
                            * heat
                            * surface_pressure
                            / temp
                            / conductivity
                            * (heat / temp / r_specific - 1)
                            / (r_specific * temp)
                        )
                        rate /= correction
                    proposal[b, i, material] = rate * dt / 4
        for b in range(masses.shape[0]):
            for g, material in state.pairs:
                candidate = np.maximum(
                    proposal[b, :, material], -masses[b, :, material]
                )
                demand = sum(np.maximum(candidate, 0) * weights[b])
                release = -sum(np.minimum(candidate, 0) * weights[b])
                fraction = (
                    min(1.0, (gases[b, g] + release) / demand)
                    if demand
                    else 1.0
                )
                transfer = np.where(
                    candidate > 0, candidate * fraction, candidate
                )
                updated = masses[b, :, material] + transfer
                actual = updated - masses[b, :, material]
                masses[b, :, material] = updated
                gases[b, g] = max(0.0, gases[b, g] - sum(actual * weights[b]))
                total[b, :, material] += actual
                energy[b, g] += (
                    sum(actual * weights[b]) * p.volume[b] * state.heat[g]
                )
    return masses, gases, total, energy


@pytest.mark.parametrize(
    "kind,volume",
    [
        ("particle_resolved", 0.25),
        ("particle_resolved", 1.0),
        ("particle_resolved", 4.0),
        ("discrete", 1.0),
    ],
)
@pytest.mark.parametrize(
    "scenario", ["uptake", "evaporation", "mixed", "disabled", "skipped"]
)
@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_native_four_substeps_match_independent_science(
    device, kind, volume, scenario, latent, prepared
):
    """Compare particle/gas changes separately and species inventory extensively."""
    from particula.gpu.kernels import condensation as module

    state = _state(device, kind, volume, scenario, latent)
    expected_m, expected_g, expected_total, expected_energy = _oracle(state)
    identities = [
        state.p.masses,
        state.p.concentration,
        state.p.charge,
        state.p.density,
        state.p.volume,
        state.g.concentration,
        state.g.molar_mass,
        state.g.partitioning,
        state.g.vapor_pressure,
    ]
    before = [a.numpy().copy() for a in identities]
    if prepared:
        bound = module._prepare_condensation_step_gpu(
            state.p, state.g, **state.kwargs
        )
        result = module._enqueue_prepared_condensation_call(bound)
    else:
        result = module.condensation_step_gpu(state.p, state.g, **state.kwargs)
    assert result[0] is state.p and result[1] is state.total
    for current, old in zip(
        [
            state.p.masses,
            state.p.concentration,
            state.p.charge,
            state.p.density,
            state.p.volume,
            state.g.concentration,
            state.g.molar_mass,
            state.g.partitioning,
            state.g.vapor_pressure,
        ],
        identities,
        strict=True,
    ):
        assert current is old
    m, g = state.p.masses.numpy(), state.g.concentration.numpy()
    npt.assert_allclose(
        m - before[0], expected_m - before[0], rtol=2e-9, atol=1e-32
    )
    npt.assert_allclose(
        g - before[5], expected_g - before[5], rtol=2e-9, atol=1e-23
    )
    npt.assert_allclose(
        state.total.numpy(), expected_total, rtol=2e-9, atol=1e-32
    )
    weights = before[1] if kind == "particle_resolved" else before[1] * volume
    for source, target in state.pairs:
        initial = (
            np.sum(before[0][:, :, target] * weights, axis=1)
            + before[5][:, source] * volume
        )
        final = (
            np.sum(m[:, :, target] * weights, axis=1) + g[:, source] * volume
        )
        npt.assert_allclose(final, initial, rtol=1e-12, atol=1e-30)
    if latent:
        npt.assert_allclose(
            state.energy.numpy(), expected_energy, rtol=2e-9, atol=1e-23
        )
    for i in (1, 2, 3, 4, 6, 7):
        npt.assert_array_equal(identities[i].numpy(), before[i])
    npt.assert_array_equal(g[:, [0, 3]], before[5][:, [0, 3]])
    npt.assert_array_equal(m[:, :, 2], before[0][:, :, 2])
    npt.assert_allclose(
        state.g.vapor_pressure.numpy(),
        np.tile(state.vp, (2, 1)),
        rtol=0,
        atol=0,
    )
    assert np.all(m >= 0) and np.all(g >= 0)


@pytest.mark.parametrize("scenario", ["uptake", "evaporation", "skipped"])
def test_native_single_material_nonidentity_map(device, scenario):
    """The optimized one-material branch reads its declared full-gas source."""
    from particula.gpu.kernels.condensation import condensation_step_gpu

    state = _state(device, scenario=scenario, single=True)
    expected = _oracle(state)
    condensation_step_gpu(state.p, state.g, **state.kwargs)
    npt.assert_allclose(
        state.p.masses.numpy(), expected[0], rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(
        state.g.concentration.numpy(), expected[1], rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(
        state.energy.numpy(), expected[3], rtol=2e-9, atol=1e-23
    )


@pytest.mark.parametrize(
    "fault",
    [
        "volume",
        "pmf_volume",
        "mass",
        "count",
        "density",
        "charge",
        "gas",
        "flags",
        "mapping",
        "storage",
        "molar",
        "output_alias",
        "primary_alias",
        "scratch_shape",
        "boolean_time",
    ],
)
def test_native_preflight_rejections_preserve_all_caller_storage(  # noqa: C901
    device, fault
):
    """Malformed state/configuration is rejected before any output writer."""
    import warp as wp

    from particula.gpu.kernels.condensation import (
        CondensationActivitySurfaceConfig,
        CondensationScratchBuffers,
        condensation_step_gpu,
    )
    from particula.gpu.species_mapping import prepare_condensation_species_map

    state = _state(
        device,
        kind="discrete" if fault == "pmf_volume" else "particle_resolved",
        volume=1.0,
    )
    fields = {
        "volume": state.p.volume,
        "pmf_volume": state.p.volume,
        "mass": state.p.masses,
        "count": state.p.concentration,
        "density": state.p.density,
        "charge": state.p.charge,
        "gas": state.g.concentration,
        "flags": state.g.partitioning,
    }
    if fault in fields:
        values = fields[fault].numpy().copy()
        value = 4.0 if fault == "pmf_volume" else -1.0
        if fault in ("volume", "charge", "gas"):
            value = np.nan
        if fault == "flags":
            value = 2
        values.flat[0] = value
        fields[fault].assign(values)
    elif fault == "mapping":
        state.kwargs["species_binding"] = None
    elif fault == "storage":
        state.p.volume = wp.ones_like(state.p.volume)
    elif fault == "molar":
        config = state.kwargs["activity_surface"]
        refs = config.molar_mass_reference.numpy()
        refs[0] *= 2
        state.kwargs["activity_surface"] = CondensationActivitySurfaceConfig(
            config.activity_mode,
            config.surface_tension_mode,
            config.water_species_index,
            config.kappas,
            wp.array(refs, dtype=wp.float64, device=device),
        )
    elif fault == "output_alias":
        state.kwargs.pop("scratch_buffers")
        state.kwargs["mass_transfer"] = state.p.masses
    elif fault == "primary_alias":
        state.g.vapor_pressure = state.g.concentration
        old_binding = state.kwargs["species_binding"]
        state.kwargs["species_binding"] = prepare_condensation_species_map(
            state.p,
            state.g,
            gas_names=old_binding.gas_names,
            species_map=old_binding.species_map,
        )
    elif fault == "scratch_shape":
        state.kwargs["scratch_buffers"] = CondensationScratchBuffers(
            positive_mass_transfer_demand=wp.zeros(
                (2, 3), dtype=wp.float64, device=device
            ),
        )
    else:
        state.kwargs["time_step"] = True
    arrays = [
        state.p.masses,
        state.p.concentration,
        state.p.density,
        state.p.volume,
        state.p.charge,
        state.g.concentration,
        state.g.vapor_pressure,
        state.g.molar_mass,
        state.g.partitioning,
        state.total,
        state.work,
        state.energy,
    ]
    before = [a.numpy().copy() for a in arrays]
    with pytest.raises(ValueError):
        condensation_step_gpu(state.p, state.g, **state.kwargs)
    for array, saved in zip(arrays, before, strict=True):
        npt.assert_array_equal(array.numpy(), saved)


def test_native_prepared_enqueue_uses_only_frozen_device_references(
    device, monkeypatch
):
    """No allocations, readbacks, synchronization, validation or container lookup."""
    import warp as wp

    from particula.gpu.kernels import condensation as module
    from particula.gpu.species_mapping import CondensationSpeciesBinding

    state = _state(device)
    expected = _oracle(state)
    prepared = module._prepare_condensation_step_gpu(
        state.p, state.g, **state.kwargs
    )
    frozen_masses, frozen_gas = state.p.masses, state.g.concentration
    replacements = []
    for carrier, names in (
        (state.p, ("masses", "concentration", "density", "charge", "volume")),
        (
            state.g,
            ("concentration", "molar_mass", "partitioning", "vapor_pressure"),
        ),
    ):
        for name in names:
            replacement = wp.zeros_like(getattr(carrier, name))
            setattr(carrier, name, replacement)
            replacements.append(replacement)
    state.p.distribution_kind = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared enqueue performed host work")

    with monkeypatch.context() as patch:
        for name in (
            "zeros",
            "zeros_like",
            "empty",
            "full",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        patch.setattr(CondensationSpeciesBinding, "validate", forbidden)
        patch.setattr(module, "_prepare_condensation_step_gpu", forbidden)
        result = module._enqueue_prepared_condensation_call(prepared)
    assert result[0] is state.p and result[1] is state.total
    npt.assert_allclose(
        frozen_masses.numpy(), expected[0], rtol=1e-12, atol=1e-30
    )
    npt.assert_allclose(frozen_gas.numpy(), expected[1], rtol=1e-12, atol=1e-30)
    for replacement in replacements:
        npt.assert_array_equal(replacement.numpy(), 0)


@pytest.mark.parametrize("no_work", ["time", "counts", "flags", "map"])
def test_native_no_work_preserves_primary_inventory(device, no_work):
    """No-work calls still refresh full gas thermodynamics and clear outputs."""
    from particula.gpu.kernels.condensation import condensation_step_gpu
    from particula.gpu.species_mapping import prepare_condensation_species_map

    state = _state(device)
    if no_work == "time":
        state.kwargs["time_step"] = 0.0
    elif no_work == "counts":
        state.p.concentration.zero_()
    elif no_work == "flags":
        state.g.partitioning.zero_()
    else:
        state.kwargs["species_binding"] = prepare_condensation_species_map(
            state.p,
            state.g,
            gas_names=state.cpu_g.name,
            species_map=CondensationSpeciesMap(tuple(state.cpu_g.name), ()),
        )
    masses, gas = (
        state.p.masses.numpy().copy(),
        state.g.concentration.numpy().copy(),
    )
    condensation_step_gpu(state.p, state.g, **state.kwargs)
    npt.assert_array_equal(state.p.masses.numpy(), masses)
    npt.assert_array_equal(state.g.concentration.numpy(), gas)
    npt.assert_array_equal(state.total.numpy(), 0)
    npt.assert_array_equal(state.energy.numpy(), 0)


@pytest.mark.parametrize(
    "activity_mode,surface_mode", [(None, 0), (0, 0), (0, 1), (1, 0), (1, 1)]
)
@pytest.mark.parametrize("narrow_gas", [False, True])
def test_native_full_material_composition_across_gas_widths(
    device, activity_mode, surface_mode, narrow_gas
):
    """Particle-only material contributes to activity/geometry with either width."""
    from dataclasses import replace

    import warp as wp

    from particula.gpu.conversion import to_warp_gas_data
    from particula.gpu.kernels import condensation as module
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
    from particula.gpu.species_mapping import prepare_condensation_species_map

    state = _state(device)
    if narrow_gas:
        selected = [1, 2]
        state.cpu_g = GasData(
            name=["B", "A"],
            molar_mass=state.cpu_g.molar_mass[selected],
            concentration=state.cpu_g.concentration[:, selected],
            partitioning=state.cpu_g.partitioning[selected],
        )
        state.g = to_warp_gas_data(state.cpu_g, device=device)
        state.pairs = ((1, 0), (0, 1))
        for name in ("vp", "heat", "accom", "diffusion"):
            setattr(state, name, getattr(state, name)[selected])
        state.flags = state.flags[:, selected]
        binding = prepare_condensation_species_map(
            state.p,
            state.g,
            gas_names=state.cpu_g.name,
            species_map=CondensationSpeciesMap(
                tuple(state.cpu_g.name), state.pairs
            ),
        )
        state.kwargs["species_binding"] = binding
        state.kwargs["thermodynamics"] = ThermodynamicsConfig(
            modes=wp.zeros(2, dtype=wp.int32, device=device),
            parameters=wp.array(
                np.column_stack((state.vp, np.zeros((2, 3)))),
                dtype=wp.float64,
                device=device,
            ),
            molar_mass_reference=wp.array(
                state.cpu_g.molar_mass, dtype=wp.float64, device=device
            ),
        )
        for name, values in (
            ("mass_accommodation", state.accom),
            ("diffusion_coefficient_vapor", state.diffusion),
            ("latent_heat", state.heat),
        ):
            state.kwargs[name] = wp.array(
                values, dtype=wp.float64, device=device
            )
        state.energy = wp.zeros((2, 2), dtype=wp.float64, device=device)
        state.kwargs["energy_transfer"] = state.energy
        state.kwargs["scratch_buffers"] = module.CondensationScratchBuffers(
            total_mass_transfer=state.total,
            work_mass_transfer=state.work,
        )
    if activity_mode is None:
        state.kwargs.pop("activity_surface")
    else:
        state.kwargs["activity_surface"] = replace(
            state.kwargs["activity_surface"],
            activity_mode=activity_mode,
            surface_tension_mode=surface_mode,
        )
    expected = _oracle(state)
    before_m, before_g = state.cpu_p.masses, state.cpu_g.concentration
    prepared = module._prepare_condensation_step_gpu(
        state.p, state.g, **state.kwargs
    )
    module._enqueue_prepared_condensation_call(prepared)
    npt.assert_allclose(
        state.p.masses.numpy() - before_m,
        expected[0] - before_m,
        rtol=2e-9,
        atol=1e-32,
    )
    npt.assert_allclose(
        state.g.concentration.numpy() - before_g,
        expected[1] - before_g,
        rtol=2e-9,
        atol=1e-23,
    )
    npt.assert_allclose(
        state.energy.numpy(), expected[3], rtol=2e-9, atol=1e-23
    )
    npt.assert_array_equal(state.p.masses.numpy()[:, :, 2], before_m[:, :, 2])


def test_native_prepared_replay_reads_current_gas_flags(device):
    """Partitioning remains live full-gas state, not a baked compact mask."""
    from particula.gpu.kernels import condensation as module

    state = _state(device)
    prepared = module._prepare_condensation_step_gpu(
        state.p, state.g, **state.kwargs
    )
    module._enqueue_prepared_condensation_call(prepared)
    masses, gas = (
        state.p.masses.numpy().copy(),
        state.g.concentration.numpy().copy(),
    )
    state.g.partitioning.zero_()
    module._enqueue_prepared_condensation_call(prepared)
    npt.assert_array_equal(state.p.masses.numpy(), masses)
    npt.assert_array_equal(state.g.concentration.numpy(), gas)
    npt.assert_array_equal(state.total.numpy(), 0)
    npt.assert_array_equal(state.energy.numpy(), 0)


def test_native_empty_particle_axis_preserves_gas(device):
    """An empty native population refreshes vapor pressure without transfers."""
    from dataclasses import replace

    from particula.gpu.conversion import to_warp_particle_data
    from particula.gpu.kernels import condensation as module
    from particula.gpu.species_mapping import prepare_condensation_species_map

    state = _state(device)
    empty = replace(
        state.cpu_p,
        masses=state.cpu_p.masses[:, :0, :].copy(),
        concentration=state.cpu_p.concentration[:, :0].copy(),
        charge=state.cpu_p.charge[:, :0].copy(),
    )
    state.p = to_warp_particle_data(empty, device=device)
    binding = prepare_condensation_species_map(
        state.p,
        state.g,
        gas_names=state.cpu_g.name,
        species_map=CondensationSpeciesMap(
            tuple(state.cpu_g.name), state.pairs
        ),
    )
    state.kwargs["species_binding"] = binding
    state.kwargs.pop("scratch_buffers")
    before = state.g.concentration.numpy().copy()
    result = module.condensation_step_gpu(state.p, state.g, **state.kwargs)
    assert result[0] is state.p and result[1].shape == (2, 0, 3)
    npt.assert_array_equal(state.g.concentration.numpy(), before)
    npt.assert_array_equal(state.energy.numpy(), 0)
