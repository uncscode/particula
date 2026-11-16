"""Sparse finite differences for local particles coupled through box gas.

A particle rate depends on that particle's composition and its box gas, not
another particle's composition. Gas derivatives follow from the same weighted
mass exchange. Coloring particle columns by material therefore needs only two
rate evaluations per material, independently of particle count. Gas columns
remain explicit finite differences. Coordinates and derivatives are scaled in
exactly the same way as CondensationODE's dense reference Jacobian.
"""

import numpy as np
from scipy.sparse import csc_matrix


def supports_local_rates(strategy):
    """Restrict coloring to known particle-local activity/surface formulas."""
    from particula.particles import activity_strategies as ac
    from particula.particles import surface_strategies as su

    return type(strategy.activity_strategy) in (
        ac.ActivityIdealMass,
        ac.ActivityIdealMolar,
        ac.ActivityIdealVolume,
        ac.ActivityKappaParameter,
    ) and type(strategy.surface_strategy) in (
        su.SurfaceStrategyMass,
        su.SurfaceStrategyMolar,
        su.SurfaceStrategyVolume,
    )


def coordinate_steps(owner, values):
    """Scale particle differences by current total mass, including dry cores.

    Initial wet coordinates can exceed dry equilibrium mass by many orders of
    magnitude. A fixed perturbation in those coordinates then probes a different
    physical regime. Use the current slot mass as the composition scale, also
    for a currently empty material. Gas dependence is linear at fixed particles.
    """
    masses = owner._base.particles.masses.copy()
    for position, index in enumerate(owner._mass_indices):
        masses[index] = values[position] * owner._scale[position]
    totals = masses.sum(axis=2)
    characteristic = np.maximum(np.abs(values), 1.0)
    for position, (box, slot, _) in enumerate(owner._mass_indices):
        characteristic[position] = totals[box, slot] / owner._scale[position]
    return np.sqrt(np.finfo(float).eps) * characteristic


def sparse_jacobian(owner, time, values):
    """Differentiate mapped continuous rates without a dense particle matrix."""
    base = owner._rate(time, values)
    delta = coordinate_steps(owner, values)
    mass_indices = owner._mass_indices
    mass_lookup = {index: column for column, index in enumerate(mass_indices)}
    gas_lookup = {
        index: owner._mass_count + column
        for column, index in enumerate(owner._gas_indices)
    }
    source_by_target = {
        (box, int(target)): int(source)
        for box, pairs in enumerate(owner._maps)
        for source, target in pairs
    }
    rows, columns, data = [], [], []
    for material in sorted({index[2] for index in mass_indices}):
        selected = [
            column
            for column, index in enumerate(mass_indices)
            if index[2] == material
        ]
        plus, minus = values.copy(), values.copy()
        plus[selected] += delta[selected]
        minus[selected] -= delta[selected]
        if owner._admissible(minus):
            difference = owner._rate(time, plus) - owner._rate(time, minus)
            divisor = 2 * delta
        else:
            difference = owner._rate(time, plus) - base
            divisor = delta
        for row, (box, slot, target) in enumerate(mass_indices):
            column = mass_lookup.get((box, slot, material))
            if column is None:
                continue
            derivative = difference[row] / divisor[column]
            rows.append(row)
            columns.append(column)
            data.append(derivative)
            source = source_by_target[(box, target)]
            gas_row = gas_lookup.get((box, source))
            if gas_row is not None:
                rows.append(gas_row)
                columns.append(column)
                data.append(
                    -owner._weights[box][slot]
                    * owner._scale[row]
                    / owner._scale[gas_row]
                    * derivative
                )
    for column in range(owner._mass_count, len(values)):
        plus, minus = values.copy(), values.copy()
        plus[column] += delta[column]
        minus[column] -= delta[column]
        if owner._admissible(minus):
            derivative = (
                owner._rate(time, plus) - owner._rate(time, minus)
            ) / (2 * delta[column])
        else:
            derivative = (owner._rate(time, plus) - base) / delta[column]
        nonzero = np.flatnonzero(derivative)
        rows.extend(nonzero)
        columns.extend([column] * len(nonzero))
        data.extend(derivative[nonzero])
    return csc_matrix((data, (rows, columns)), shape=(len(values), len(values)))
