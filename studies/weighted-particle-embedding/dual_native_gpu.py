"""Use native mapped condensation with a study-local ideal-molar activity
binding.
"""

from dataclasses import replace

import numpy as np
import warp as wp
from dual_materials import (
    SURFACE_DENSITY,
    SURFACE_TENSION,
    TEMPERATURE,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gpu.conversion import to_warp_gas_data, to_warp_particle_data
from particula.gpu.kernels.condensation import (
    CondensationActivitySurfaceConfig,
    _enqueue_prepared_condensation_call,
    _prepare_condensation_step_gpu,
)
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
from particula.gpu.species_mapping import prepare_condensation_species_map
from vapor_properties import materials


@wp.kernel
def ideal_activities(
    mass: wp.array3d(dtype=wp.float64),
    molar: wp.array(dtype=wp.float64),
    values: wp.array3d(dtype=wp.float64),
):
    """Supply neutral material mole fractions for each active particle."""
    b, i = wp.tid()
    total = wp.float64(0.0)
    for s in range(molar.shape[0]):
        total += mass[b, i, s] / molar[s]
    for s in range(molar.shape[0]):
        result = wp.float64(0.0)
        if total > wp.float64(0.0):
            result = mass[b, i, s] / molar[s] / total
        values[b, i, s] = result


class IdealActivities:
    """Supply the same mole fractions as native CPU ActivityIdealMolar."""

    def __init__(self, shape, molar, device):
        """Allocate the device activity workspace."""
        self.values = wp.zeros(shape, dtype=wp.float64, device=device)
        self.molar = molar
        self.device = device

    def evaluate(self, masses, concentration, temperature, failure, substep):
        """Refresh activities from the current native particle state."""
        wp.launch(
            ideal_activities,
            dim=self.values.shape[:2],
            inputs=[masses, self.molar, self.values],
            device=self.device,
        )


class DirectDualCondensation:
    """Retain native transfer/budget kernels and graph-capture their public step
    body.
    """

    def __init__(self, state, dt, properties=None):
        """Bind each volatile lane to its corresponding particle material."""
        spec = materials(2) if properties is None else properties
        lanes = spec.species
        size = len(spec.molar)
        self.device = wp.get_device("cuda:0")
        self.host = state
        self.particles = to_warp_particle_data(
            state.particles, device=str(self.device)
        )
        self.gas = to_warp_gas_data(state.gas, device=str(self.device))

        def array(x):
            return wp.array(x, dtype=wp.float64, device=self.device)

        params = np.zeros((lanes, 4))
        params[:, 0] = spec.pressure
        therm = ThermodynamicsConfig(
            wp.zeros(lanes, dtype=wp.int32, device=self.device),
            array(params),
            array(spec.molar[:lanes]),
        )
        self.provider = IdealActivities(
            self.particles.masses.shape, array(spec.molar), self.device
        )
        activity = CondensationActivitySurfaceConfig(
            3,
            0,
            0,
            array(np.zeros(size)),
            self.provider.molar,
            provider=self.provider,
        )
        binding = prepare_condensation_species_map(
            self.particles,
            self.gas,
            gas_names=spec.names,
            species_map=CondensationSpeciesMap(
                spec.names, tuple((s, s) for s in range(lanes))
            ),
        )
        call = _prepare_condensation_step_gpu(
            self.particles,
            self.gas,
            TEMPERATURE,
            101325.0,
            dt,
            array(np.full(size, SURFACE_TENSION)),
            array(spec.accommodation),
            array(spec.diffusivity),
            thermodynamics=therm,
            activity_surface=activity,
            species_binding=binding,
        )
        self.call = replace(call, kelvin_density=SURFACE_DENSITY)

    def capture(self, group=1):
        """Capture the native four-substep transfer sequence."""
        with wp.ScopedCapture(
            device=self.device, force_module_load=True
        ) as recorder:
            for _ in range(group):
                _enqueue_prepared_condensation_call(self.call)
        self.graph = recorder.graph
        self.group = group

    def run(self, steps):
        """Advance consecutive closed-gas steps on the device."""
        if steps % self.group:
            raise ValueError("Step count does not match the captured group")
        for _ in range(steps // self.group):
            wp.capture_launch(self.graph)

    def update_gas(self, concentration):
        """Replace the gas concentrations at a prescribed boundary."""
        self.gas.concentration.assign(concentration)

    def snapshot(self):
        """Return native masses and gas concentrations after synchronization."""
        if self.call.failure.numpy()[0]:
            raise ArithmeticError(
                "Native dual-vapor provider reported a failure"
            )
        self.host.particles.masses[:] = self.particles.masses.numpy()
        self.host.particles.concentration[:] = (
            self.particles.concentration.numpy()
        )
        self.host.gas.concentration[:] = self.gas.concentration.numpy()
        return self.host

    def close(self):
        """Wait for any outstanding native work."""
        wp.synchronize_device(self.device)
