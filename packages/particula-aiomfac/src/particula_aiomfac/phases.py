"""Liquid stability and coexistence on an analytical formula-unit basis."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import expit

from ._equilibrium import ActivityDomainError, ConvergenceError
from .model import ActivityResult


@dataclass(frozen=True)
class LiquidEquilibrium:
    """Phase masses in the input mass unit and dimensionless diagnostics."""

    alpha_masses: np.ndarray
    beta_masses: np.ndarray | None
    alpha: ActivityResult
    beta: ActivityResult | None
    chemical_potential_error: float
    minimum_tpd: float


def component_molar_masses(model):
    """Analytical kg/mol in declaration order, excluding virtual species."""
    mw = np.empty(model.ncomponents + model.virtual_count)
    mw[model.neutral_indices] = model.neutral_mw
    mw[model.salt_indices] = model.salt_mw
    return mw[: model.ncomponents]


class _LiquidSystem:
    def __init__(self, model, mass, temperature, *, evaluator=None):
        self.model = model
        self.evaluator = model if evaluator is None else evaluator
        self.temperature = temperature
        self.present = np.flatnonzero(mass > 0)
        self.mw = component_molar_masses(model)
        moles = mass[self.present] / self.mw[self.present]
        self.feed = moles / moles.sum()

    def potentials(self, x):
        """Chemical potentials divided by RT up to conserved linear constants."""
        p = self.model
        mass = np.zeros((*x.shape[:-1], p.ncomponents))
        mass[..., self.present] = x * self.mw[self.present]
        r = self.evaluator.evaluate(mass, self.temperature)
        mu = np.zeros((*x.shape[:-1], p.ncomponents + p.virtual_count))
        with np.errstate(divide="ignore", invalid="ignore"):
            neutral = (
                np.log(r.neutral_mole_fractions) + r.neutral_log_coefficients
            )
            if p.carbonate and p.carbonate_model == "pinned":
                # The closed reaction constants use CO2 on the molality scale.
                n = r.species_moles_per_kg_input[..., : len(p.neutrals)]
                neutral[..., p.co2] = (
                    np.log(n[..., p.co2] / (n @ p.neutral_mw))
                    + r.neutral_log_coefficients[..., p.co2]
                )
            mu[..., p.neutral_indices] = neutral
            ions = np.log(r.ion_molalities) + r.ion_log_coefficients
            for lane, stoich in zip(p.salt_indices, p.stoich, strict=True):
                used = stoich > 0
                mu[..., lane] = ions[..., used] @ stoich[used]
        result = mu[..., self.present]
        if not np.all(np.isfinite(result)):
            raise ActivityDomainError(
                "undefined chemical potential in a present component"
            )
        return result


def _seeds(feed):
    """Composition-enriched directions including the feed and every vertex."""
    yield feed
    for vertex in np.eye(len(feed)):
        yield 0.9 * vertex + 0.1 * feed


def _stability(system, reference):
    # Softmax coordinates keep each trial inside the occupied composition face.
    # A common offset in the component reference cancels from the gradient.
    minima = []
    for seed in _seeds(system.feed):
        # Concentrated seeds can exceed the inner speciation solver's domain.
        # Backtrack towards the verified feed, preserving the search direction.
        for _ in range(12):
            try:
                system.potentials(seed)
                break
            except ActivityDomainError:
                seed = (system.feed + seed) / 2
        else:
            raise ConvergenceError(
                "could not initialize a liquid stability direction"
            )

        def objective(logits):
            weights = np.exp(np.r_[logits, 0.0] - max(0.0, np.max(logits)))
            x = weights / weights.sum()
            delta = system.potentials(x) - reference
            tpd = float(x @ delta)
            return tpd, (x * (delta - tpd))[:-1]

        fit = minimize(
            objective,
            np.log(seed[:-1] / seed[-1]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-32.0, 32.0)] * (len(seed) - 1),
            options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 150, "maxls": 50},
        )
        if not fit.success and np.linalg.norm(fit.jac, ord=np.inf) > 1e-7:
            raise ConvergenceError(
                "liquid stability search failed: " + fit.message
            )
        weights = np.exp(np.r_[fit.x, 0.0] - max(0.0, np.max(fit.x)))
        minima.append((float(fit.fun), weights / weights.sum()))
    return sorted(minima, key=lambda item: item[0])


def phase_equilibrium(model, masses, temperature, water_index):
    """Resolve one homogeneous or two coexisting liquids at fixed inventory."""
    mass = np.asarray(masses, dtype=float)
    if mass.shape != (model.ncomponents,):
        raise ValueError("phase_equilibrium expects one analytical inventory")
    direct = model.evaluate(mass, temperature, diagnostics=True)
    if model.carbonate and 221 in model.ions and model.sulfate:
        if direct.diagnostics["precipitated_caso4_mol_per_kg_input"] > 0:
            raise ActivityDomainError(
                "stable liquid calculation requires a solid CaSO4 phase"
            )
    from .reciprocal import reciprocal_equilibrium, requires_ion_basis

    if requires_ion_basis(model, mass):
        return reciprocal_equilibrium(
            model, mass, temperature, water_index, direct
        )
    system = _LiquidSystem(model, mass, temperature)
    feed = system.feed
    if len(feed) == 1:
        return LiquidEquilibrium(mass.copy(), None, direct, None, 0.0, 0.0)
    reference = system.potentials(feed)
    unstable = _stability(system, reference)
    if unstable[0][0] >= -1e-8:
        return LiquidEquilibrium(
            mass.copy(), None, direct, None, 0.0, unstable[0][0]
        )

    candidates = []
    for distance, seed in unstable:
        if distance >= -1e-8:
            break
        # f is the fraction of each conserved component transferred to alpha.
        initial = 0.7 * np.min(feed / seed) * seed / feed

        def split(f):
            amounts = np.stack((feed * f, feed * (1 - f)))
            potentials = system.potentials(amounts)
            return float(np.sum(amounts * potentials)), potentials[
                0
            ] - potentials[1]

        def gibbs(f):
            energy, difference = split(f)
            return energy, feed * difference

        # A newly appearing liquid can occupy far less than its limiting
        # lever-rule fraction. Follow the negative tangent until the *finite*
        # split actually lowers Gibbs energy, avoiding the homogeneous root.
        for _ in range(16):
            if split(initial)[0] < feed @ reference:
                break
            initial *= 0.2
        fit = minimize(
            gibbs,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(1e-13, 1 - 1e-13)] * len(feed),
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 250},
        )
        logits = np.log(fit.x / (1 - fit.x))
        root = least_squares(
            lambda v: split(expit(v))[1],
            logits,
            bounds=(-32.0, 32.0),
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
        )
        fraction = expit(root.x)
        energy, residual = split(fraction)
        err = np.max(np.abs(residual))
        if err < 1e-7 and energy < feed @ reference and np.ptp(fraction) > 1e-5:
            candidates.append((energy, fraction, err))
    if not candidates:
        raise ConvergenceError(
            "unstable liquid without a verified coexistence solution"
        )
    _, fraction, error = min(candidates, key=lambda item: item[0])
    minimum = _stability(system, system.potentials(feed * fraction))[0][0]
    if minimum < -1e-7:
        raise ConvergenceError(
            "two-liquid solution unstable to an additional phase"
        )
    alpha = np.zeros_like(mass)
    alpha[system.present] = mass[system.present] * fraction
    beta = mass - alpha
    if alpha[water_index] / alpha.sum() < beta[water_index] / beta.sum():
        alpha, beta = beta, alpha
    return LiquidEquilibrium(
        alpha,
        beta,
        model.evaluate(alpha, temperature, diagnostics=True),
        model.evaluate(beta, temperature, diagnostics=True),
        float(error),
        minimum,
    )
