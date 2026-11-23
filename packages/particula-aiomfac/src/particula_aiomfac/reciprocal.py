"""Liquid stability with independently transferable, charge-balanced ions."""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, least_squares, minimize
from scipy.special import expit, softmax

from ._equilibrium import ActivityDomainError, ConvergenceError
from .model import _tables


def ion_basis(model):
    """Return an ion-input model and source-to-constituent mass fractions.

    Columns are native neutrals followed by individual ions in model order.
    For active bisulfate chemistry the conserved analytical input uses H+ and
    sulfate. Each liquid internally re-equilibrates bisulfate at every trial.
    """
    n = len(model.neutrals)
    ions = np.asarray(model.ions, dtype=int)
    cations = model.ncation
    tables = _tables()
    mw = (
        np.r_[
            tables["SMWC"][ions[:cations] - 201],
            tables["SMWA"][ions[cations:] - 241],
        ]
        * 0.001
    )
    stoich = model.stoich.copy()
    if model.sulfate:
        hs, proton, sulfate = map(model.ions.index, (248, 205, 261))
        stoich[:, proton] += stoich[:, hs]
        stoich[:, sulfate] += stoich[:, hs]
        stoich[:, hs] = 0.0
    if model.carbonate and model.carbonate_model == "gibbs_consistent":
        hc, proton, co = map(model.ions.index, (250, 205, 262))
        stoich[:, proton] += stoich[:, hc]
        stoich[:, co] += stoich[:, hc]
        stoich[:, hc] = 0.0
    width = n + len(ions)
    expanded = replace(
        model,
        ncomponents=width,
        virtual_count=0,
        neutral_indices=np.arange(n),
        salt_indices=n + np.arange(len(ions)),
        salt_mw=mw,
        stoich=np.eye(len(ions)),
    )
    matrix = np.zeros((model.ncomponents, width))
    for j, lane in enumerate(model.neutral_indices):
        if lane < model.ncomponents:
            matrix[lane, j] = 1.0
    for j, lane in enumerate(model.salt_indices):
        if lane < model.ncomponents:
            weights = stoich[j] * mw
            matrix[lane, n:] = weights / weights.sum()
    return expanded, matrix


def requires_ion_basis(model, mass):
    """Determine whether a formula-unit basis restricts feasible transfers."""
    real = model.salt_indices < model.ncomponents
    active = model.stoich[real][mass[model.salt_indices[real]] > 0]
    positive = np.any(active > 0, axis=0)
    if (
        model.carbonate
        and model.carbonate_model == "pinned"
        and (
            positive[model.ions.index(250)]
            or positive[model.ions.index(262)]
            or (
                model.neutral_indices[model.co2] < model.ncomponents
                and mass[model.neutral_indices[model.co2]] > 0
            )
        )
    ):
        raise ActivityDomainError(
            "stable multiphase carbonate requires Gibbs-consistent CO2 "
            "and ion activities; use single_liquid for carbonate speciation"
        )
    if model.carbonate and model.carbonate_model == "gibbs_consistent":
        declared_co2 = model.neutral_indices[model.co2]
        has_co2 = declared_co2 < model.ncomponents and mass[declared_co2] > 0
        if positive.any() and (
            has_co2
            or positive[model.ions.index(250)]
            or positive[model.ions.index(262)]
        ):
            return True
    if model.sulfate and 248 in model.ions and positive[model.ions.index(248)]:
        return True
    return bool(
        positive[: model.ncation].sum() > 1
        and positive[model.ncation :].sum() > 1
    )


def hydrate_carbon(model, matrix, mass, expanded):
    """Build the charge-balanced component basis at this particular inventory.

    Each mole of declared CO₂ consumes one mole of water to supply one
    carbonate and two protons. Both input mass labels then follow the
    corresponding physical ion mass fractions. At fixed total inventory this
    is a change of components; the inner chemical solver determines actual
    CO₂, bicarbonate, carbonate, proton and hydroxide anew in each phase.
    """
    h, carbonate, oh = map(model.ions.index, (205, 262, 247))
    n = len(model.neutrals)
    water = model.neutral_indices[model.water]
    co2 = model.neutral_indices[model.co2]
    moles_co2 = (
        mass[co2] / model.neutral_mw[model.co2] if co2 < len(mass) else 0.0
    )
    transfer = np.zeros(matrix.shape[1])
    if moles_co2 > 0:
        water_moles = moles_co2
        donor = model.co2
        transfer[[n + h, n + carbonate]] = expanded.salt_mw[[h, carbonate]] * [
            2,
            1,
        ]
    else:
        # A basic carbonate feed without an acid carrier must still freely
        # allocate carbon and its counterions. Use the equivalent components
        # CO₂ and OH⁻: CO₃²⁻ + H₂O → CO₂ + 2 OH⁻.
        initial = mass @ matrix
        if initial[n + carbonate] == 0 or initial[n + h] > 0:
            return matrix
        water_moles = initial[n + carbonate] / expanded.salt_mw[carbonate]
        donor = n + carbonate
        transfer[model.co2] = model.neutral_mw[model.co2]
        transfer[n + oh] = 2 * expanded.salt_mw[oh]
    water_fraction = water_moles * model.neutral_mw[model.water] / mass[water]
    if water_fraction >= 1.0:
        raise ActivityDomainError(
            "aqueous carbonate allocation requires sufficient hydration water"
        )
    transfer /= transfer.sum()
    matrix = matrix.copy()
    contribution = matrix[:, donor].copy()
    matrix[:, donor] = 0.0
    contribution[water] += water_fraction
    matrix[water, model.water] -= water_fraction
    matrix += contribution[:, None] * transfer
    return matrix


class _ChargeBalancedLiquids:
    """Parameterize electroneutral liquid compositions and phase transfers."""

    def __init__(self, model, mass, temperature):
        self.model, self.matrix = ion_basis(model)
        if model.carbonate and model.carbonate_model == "gibbs_consistent":
            self.matrix = hydrate_carbon(model, self.matrix, mass, self.model)
        self.mass = mass
        self.temperature = temperature
        self.physical = mass @ self.matrix
        self.active = self.physical > 0.0
        self.mw = np.r_[self.model.neutral_mw, self.model.salt_mw][self.active]
        self.z = np.r_[np.zeros(len(model.neutrals)), model.charge][self.active]
        amounts = self.physical[self.active] / self.mw
        self.feed = amounts / amounts.sum()
        if abs(self.z @ self.feed) > 5e-12:
            raise ActivityDomainError(
                "expanded analytical ion inventory is not neutral"
            )
        # One potential reference fixes the arbitrary liquid electric potential.
        self.pivot = int(np.argmax(abs(self.z) * self.feed))
        self.solvent = int(np.argmax(np.where(self.z == 0.0, self.feed, -1.0)))
        self.free = np.arange(len(self.z)) != self.pivot
        self.composition = self.free & (np.arange(len(self.z)) != self.solvent)

    def chemistry(self, amounts, *, diagnostics=False):
        mass = np.zeros((*amounts.shape[:-1], len(self.active)))
        mass[..., self.active] = amounts * self.mw
        return self.model.evaluate(
            mass, self.temperature, diagnostics=diagnostics
        )

    def potentials(self, x):
        r = self.chemistry(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.concatenate(
                (
                    np.log(r.neutral_mole_fractions)
                    + r.neutral_log_coefficients,
                    np.log(r.ion_molalities) + r.ion_log_coefficients,
                ),
                axis=-1,
            )[..., self.active]
        if not np.isfinite(mu).all():
            raise ActivityDomainError(
                "undefined individual-ion chemical potential"
            )
        return mu

    def coordinate(self, logs, mask):
        # Gauge-fix logits without changing their decoded physical allocation.
        logs = logs.copy()
        if mask is self.composition:
            logs -= logs[self.solvent]
        return (logs - self.z * (logs[self.pivot] / self.z[self.pivot]))[mask]

    def balanced(self, v, mask, *, both=False):
        full = np.zeros_like(self.feed)
        full[mask] = v
        if mask is self.composition:

            def decode(voltage):
                return softmax(full + voltage * self.z)

            weights = self.z
        else:

            def decode(voltage):
                return expit(full + voltage * self.z)

            weights = self.feed * self.z
        # The root is unique since d(sum z_i n_i)/d(voltage) is positive.
        potential = brentq(
            lambda u: weights @ decode(u), -260.0, 260.0, xtol=1e-14
        )
        if both:
            return np.stack(
                (decode(potential), expit(-full - potential * self.z))
            )
        return decode(potential)

    def starts(self):
        yield self.feed
        for j in np.flatnonzero(self.z == 0):
            vertex = np.zeros_like(self.feed)
            vertex[j] = 1.0
            yield 0.9 * vertex + 0.1 * self.feed
        for c in np.flatnonzero(self.z > 0):
            for a in np.flatnonzero(self.z < 0):
                vertex = np.zeros_like(self.feed)
                vertex[[c, a]] = [-self.z[a], self.z[c]]
                yield 0.7 * (vertex / vertex.sum()) + 0.3 * self.feed

    def stability(self, reference, *, anchor=None):
        def objective(v):
            x = self.balanced(v, self.composition)
            try:
                difference = self.potentials(x) - reference
            except ActivityDomainError:
                # Reject an inadmissible trial in the Wolfe line search.
                return np.inf, np.zeros_like(v)
            tpd = x @ difference
            variance = x @ self.z**2
            slope = np.dot(x * self.z, difference) / variance
            jac = x * (difference - tpd - self.z * slope)
            return tpd, jac[self.composition]

        minima = []
        anchor = self.feed if anchor is None else anchor / anchor.sum()
        for seed in self.starts():
            for _ in range(8):
                start = self.coordinate(np.log(seed), self.composition)
                fit = minimize(
                    objective,
                    start,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(-70.0, 70.0)] * len(start),
                    options={
                        "maxiter": 220,
                        "ftol": 1e-14,
                        "gtol": 1e-9,
                        "maxls": 50,
                    },
                )
                if np.isfinite(fit.fun) and np.max(abs(fit.jac)) <= 1e-7:
                    break
                # A starting direction can leave the inner speciation domain.
                # Reduce its amplitude while retaining the enrichment direction.
                seed = (anchor + seed) / 2.0
            else:
                raise ConvergenceError(
                    "independent-ion stability search failed"
                )
            minima.append(
                (float(fit.fun), self.balanced(fit.x, self.composition))
            )
        return sorted(minima, key=lambda pair: pair[0])

    def split(self, start):
        def values(v):
            f = self.balanced(v, self.free, both=True)
            phases = self.feed * f
            mu = self.potentials(phases)
            return np.sum(mu * phases), mu[0] - mu[1], f

        def objective(v):
            energy, diff, f = values(v)
            weight = self.feed * f[0] * f[1]
            slope = np.dot(weight * self.z, diff) / (weight @ self.z**2)
            return energy, (weight * (diff - self.z * slope))[self.free]

        def residual(v):
            _, diff, _ = values(v)
            return (diff - self.z * (diff[self.pivot] / self.z[self.pivot]))[
                self.free
            ]

        limit = 42.0
        v0 = self.coordinate(np.log(start / (1.0 - start)), self.free)
        fit = minimize(
            objective,
            v0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-limit, limit)] * len(v0),
            options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 300, "maxls": 50},
        )
        root = least_squares(
            residual,
            fit.x,
            bounds=(-limit, limit),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        energy, _, fraction = values(root.x)
        return energy, fraction, float(np.max(abs(root.fun)))


def reciprocal_equilibrium(model, mass, temperature, water_index, direct):
    """Resolve liquid phases conserving each neutral and ionic component."""
    from .phases import LiquidEquilibrium

    sys = _ChargeBalancedLiquids(model, mass, temperature)
    reference = sys.potentials(sys.feed)
    minima = sys.stability(reference)
    if minima[0][0] >= -1e-8:
        return LiquidEquilibrium(
            mass.copy(), None, direct, None, 0.0, minima[0][0]
        )
    solutions = []
    tried = []
    for tpd, seed in minima:
        if tpd >= -1e-8:
            break
        if any(np.max(abs(prior - seed)) < 1e-6 for prior in tried):
            continue
        tried.append(seed)
        start = 0.7 * np.min(sys.feed / seed) * seed / sys.feed
        energy, f, error = sys.split(start)
        if (
            error <= 1e-7
            and np.ptp(f[0]) > 1e-5
            and energy < sys.feed @ reference
        ):
            solutions.append((energy, f, error))
    if not solutions:
        raise ConvergenceError("no verified charge-balanced liquid split")
    _, fraction, error = min(solutions, key=lambda candidate: candidate[0])
    amounts = sys.feed * fraction
    minimum = sys.stability(sys.potentials(amounts[0]), anchor=amounts[0])[0][0]
    if minimum < -1e-7:
        raise ConvergenceError(
            "reciprocal liquid solution unstable to an additional phase"
        )

    f = np.zeros_like(sys.active, dtype=float)
    f[sys.active] = fraction[0]
    alpha = mass * (sys.matrix @ f)
    beta = mass - alpha
    if alpha[water_index] / alpha.sum() < beta[water_index] / beta.sum():
        alpha, beta = beta, alpha
        amounts = amounts[::-1]
    # Source labels attribute conserved mass; true liquid composition comes
    # from the separately equilibrated primitive ion and neutral inventories.
    alpha_chem = sys.chemistry(amounts[0], diagnostics=True)
    beta_chem = sys.chemistry(amounts[1], diagnostics=True)
    if model.carbonate and model.carbonate_model == "gibbs_consistent":
        # Verify transfer equilibrium for every reaction product, including
        # CO₂, even when its source inventory was introduced as a salt.
        valid = alpha_chem.neutral_defined & beta_chem.neutral_defined
        delta = np.log(
            alpha_chem.neutral_activities[valid]
            / beta_chem.neutral_activities[valid]
        )
        ions = alpha_chem.ion_defined & beta_chem.ion_defined
        z = model.charge[ions]
        ion_delta = (
            np.log(
                alpha_chem.ion_molalities[ions] / beta_chem.ion_molalities[ions]
            )
            + alpha_chem.ion_log_coefficients[ions]
            - beta_chem.ion_log_coefficients[ions]
        )
        ion_delta -= z * ((ion_delta @ z) / (z @ z))
        error = max(
            error,
            np.max(abs(delta), initial=0),
            np.max(abs(ion_delta), initial=0),
        )
        if not np.isfinite(error) or error > 1e-7:
            raise ConvergenceError(
                "reactive species did not reach interphase electrochemical equilibrium"
            )
    return LiquidEquilibrium(alpha, beta, alpha_chem, beta_chem, error, minimum)
