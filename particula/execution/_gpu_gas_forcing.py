"""Selective gas forcing without returning evolving gas state to the host."""

from collections.abc import Mapping

import numpy as np
import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatVector,
    IntVector,
)
from particula.execution.prepared_updates import forcing_array
from particula.gpu.kernels.thermodynamics import _buck_vapor_pressure
from particula.util.constants import GAS_CONSTANT

_R = wp.constant(wp.float64(GAS_CONSTANT))


@wp.kernel
def _evaluate_targets(
    values: FloatMatrix,
    columns: IntVector,
    modes: IntVector,
    pressures: FloatVector,
    molar_mass: FloatVector,
    temperature: FloatVector,
    invalid: IntVector,
):
    """Convert prescribed ratios to kg/m3 using device temperature."""
    box, lane = wp.tid()  # type: ignore[misc]
    value = values[box, lane]
    if modes[lane] >= 0:
        pressure = pressures[lane]
        if modes[lane] == 1:
            pressure = _buck_vapor_pressure(temperature[box])
        value = (
            value
            * pressure
            * molar_mass[columns[lane]]
            / (_R * temperature[box])
        )
    if not wp.isfinite(value) or value < wp.float64(0.0):
        wp.atomic_max(invalid, 0, 1)
    values[box, lane] = value


@wp.kernel
def _commit_targets(values: FloatMatrix, columns: IntVector, gas: FloatMatrix):
    """Overwrite only the explicitly prescribed species."""
    box, lane = wp.tid()  # type: ignore[misc]
    gas[box, columns[lane]] = values[box, lane]


def apply_species_forcing(  # noqa: C901
    owner, vapor, concentrations, saturation_ratios
):
    """Validate all named targets before changing any physical state."""
    rows = []
    names = owner.gas_names
    boxes = owner._session.dimensions.n_boxes
    seen = set()
    for targets, saturation in (
        (concentrations, False),
        (saturation_ratios, True),
    ):
        if targets is None:
            continue
        if not isinstance(targets, Mapping):
            raise TypeError("gas targets must be mappings from species names")
        for name, value in targets.items():
            if not isinstance(name, str) or name not in names:
                raise ValueError(f"unknown gas species: {name!r}")
            if name in seen:
                raise ValueError("supply only one target for each gas species")
            seen.add(name)
            column = names.index(name)
            mode, pressure = vapor[column] if saturation else (-1, 0.0)
            if saturation and mode == 0 and pressure <= 0:
                raise ValueError(
                    "saturation forcing requires positive vapor pressure"
                )
            values = forcing_array(value, (boxes,), name, positive=False)
            rows.append((column, mode, pressure, values))
    if not rows:
        raise ValueError("supply at least one named gas target")
    owner.synchronize()
    with owner._scope():
        columns = wp.array(
            [r[0] for r in rows], dtype=wp.int32, device=owner.device.native
        )
        modes = wp.array(
            [r[1] for r in rows], dtype=wp.int32, device=owner.device.native
        )
        pressures = wp.array(
            [r[2] for r in rows], dtype=wp.float64, device=owner.device.native
        )
        values = wp.array(
            np.column_stack([r[3] for r in rows]),
            dtype=wp.float64,
            device=owner.device.native,
        )
        invalid = wp.zeros(1, dtype=wp.int32, device=owner.device.native)
        session = owner._session
        wp.launch(
            _evaluate_targets,
            dim=values.shape,
            inputs=[
                values,
                columns,
                modes,
                pressures,
                session.gas.molar_mass,
                session.environment.temperature,
                invalid,
            ],
            device=owner.device.native,
        )
        owner._wait()
        if invalid.numpy()[0]:
            raise ValueError("gas forcing produced a nonphysical concentration")
        try:
            wp.launch(
                _commit_targets,
                dim=values.shape,
                inputs=[values, columns, session.gas.concentration],
                device=owner.device.native,
            )
            owner._wait()
        except BaseException:
            owner._fault()
            raise
