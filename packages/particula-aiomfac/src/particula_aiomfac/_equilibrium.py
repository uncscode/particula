"""Closed-liquid v3.15 equilibria. Trial inventories are mol/kg analytical input."""

import numpy as np
from scipy.optimize import brentq, root

from ._terms import electrolyte_terms, short_range


CARBONATE_EVALUATIONS_PER_DIMENSION = 500


class ActivityDomainError(ValueError):
    """An indexed composition or equilibrium failure during evaluation."""

    def __init__(self, reason, batch_index=None):
        self.reason = reason
        self.batch_index = batch_index
        super().__init__(
            reason
            if batch_index is None
            else f"{reason} at batch index {batch_index}"
        )


class ConvergenceError(ActivityDomainError):
    """A physical root could not be verified at the reported batch index."""


def log_constants(t):
    """Natural-log constants from SubModDissociationEquil, in upstream order."""
    ts = np.clip(t, 180.0, 473.0)
    t0 = 298.15
    cp, slope = -205.594443486, -0.994992864240
    enthalpy = -18554.748598 - cp * t0 + 0.5 * slope * t0 * t0
    ks = (
        -4.54916799587
        - (
            enthalpy * (1 / ts - 1 / t0)
            - (cp - slope * t0) * np.log(ts / t0)
            - 0.5 * slope * (ts - t0)
        )
        / 8.314472
    )
    k1 = (
        -820.4327
        - 0.14027266 * t
        + 50275.49 / t
        + 126.8339 * np.log(t)
        - 3879660 / t**2
    )
    k2 = (
        -248.4192
        - 0.07489962 * t
        + 11862.43 / t
        + 38.92561 * np.log(t)
        - 1297999 / t**2
    )
    kw = -148.16780 + 0.8933802 * t - 0.002332199 * t * t + 2.146860e-6 * t**3
    return np.array([k1, k2, kw, ks])


def trial_logs(p, neutral, ion, t):
    """Return molal ion log coefficients, log water activity and solvent kg."""
    if p.carbonate and p.carbonate_model == "gibbs_consistent":
        from .carbonate import activity_terms

        sr, mr, lr, _, conversion, _, mass = activity_terms(
            p, neutral[None, :], ion[None, :], np.array([t])
        )
        gamma = (sr + mr + lr)[0]
        xw = neutral[p.water] / (neutral.sum() + ion.sum())
        return (
            gamma[len(neutral) :] - conversion[0],
            np.log(xw) + gamma[p.water],
            mass[0],
        )
    solvent = neutral @ p.neutral_mw
    m = ion / solvent
    x = np.r_[neutral, ion] / (neutral.sum() + ion.sum())
    sr = short_range(p, x[None, :], t)[0]
    mr, lr, _, conversion = electrolyte_terms(
        p, (neutral / neutral.sum())[None, :], m[None, :], np.array([t])
    )
    log_gamma = sr + mr[0] + lr[0]
    return (
        log_gamma[len(neutral) :] - conversion[0],
        log_gamma[p.water] + np.log(x[p.water]),
        solvent,
    )


def solve(p, neutral, ion, t):
    """Update one private trial row only after all applicable equations pass."""
    n0, i0 = neutral.copy(), ion.copy()
    idx = p.ions.index
    h = idx(205)
    coupled = p.sulfate
    if coupled:
        hs, so = idx(248), idx(261)
        sulfur = i0[hs] + i0[so]
    else:
        sulfur = 0.0
    carbon = 0.0
    if p.carbonate:
        oh, hc, co = map(idx, (247, 250, 262))
        carbon = i0[hc] + i0[co] + n0[p.co2]
    initial_solvent = n0 @ p.neutral_mw
    initial_cations = i0[: p.ncation].sum()
    initial_sulfur = sulfur
    solid = 0.0
    # The closed carbonate routine removes stoichiometric CaSO4 before solving.
    # Its trace floor is defined per kg of the initial electrolyte-free solvent.
    extension = p.carbonate and p.carbonate_model == "gibbs_consistent"
    if p.carbonate and not extension and sulfur and 221 in p.ions:
        calcium = idx(221)
        remaining = (
            sulfur - i0[calcium]
            if i0[calcium] < sulfur
            else min(
                1e-5 * sulfur, 1000 * np.finfo(float).eps * initial_solvent
            )
        )
        solid = sulfur - remaining
        i0[calcium] = max(0.0, i0[calcium] - sulfur) + (
            remaining if i0[calcium] >= sulfur else 0.0
        )
        i0[h] += i0[hs]
        i0[hs], i0[so] = 0.0, remaining
        sulfur = remaining
        ion[:] = i0
    constants = log_constants(t)
    checks = np.zeros(
        4
    )  # largest mass action residual, charge, carbon and sulfur
    calls = retries = 0
    if carbon > (0.0 if extension else np.finfo(float).eps * initial_solvent):
        # Electroneutrality and carbon conservation close the 5/6 variable system.
        # Scale the conservation rows by inventory for trace carbon.
        coupled = sulfur > 0
        passive = i0.copy()
        passive[[oh, hc, co, h]] = 0
        if coupled:
            passive[[hs, so]] = 0
        spectator = passive @ p.charge
        charge_scale = max(
            abs(spectator), carbon, sulfur, i0[oh], i0[h], 1e-300
        )
        solvent0 = initial_solvent
        water_available = min(3.5 * initial_cations, 0.98 * n0[p.water])
        hydroxide_max = water_available + i0[oh]
        proton_max = (
            i0[h]
            + i0[hc]
            + 2 * n0[p.co2]
            + water_available
            + (i0[hs] if p.sulfate else 0.0)
        )
        limits = np.array(
            [
                min(proton_max, carbon),
                carbon,
                min(0.5 * proton_max, carbon),
                hydroxide_max,
                proton_max,
            ]
        )

        def residual(log_amount):
            nonlocal calls
            calls += 1
            amounts = np.exp(np.clip(log_amount, -700, 50))
            b, c, d, o, hydrogen = amounts[:5]
            ion[[hc, co, oh, h]] = b, c, o, hydrogen
            neutral[p.water] = n0[p.water] + d - n0[p.co2] - (o - i0[oh])
            neutral[p.co2] = d
            if coupled:
                # A logit keeps both sulfur species positive up to roundoff.
                ratio = amounts[5]
                ion[hs], ion[so] = (
                    sulfur * ratio / (1 + ratio),
                    sulfur / (1 + ratio),
                )
            if neutral[p.water] <= 0:
                return np.full(len(amounts), 1e3 + abs(neutral[p.water]))
            g, water, solv = trial_logs(p, neutral, ion, t)
            l = np.log(amounts[:5]) - np.log(solv)
            lb, lc, ld, lo, lh = l
            gd = 2 * (ion @ p.co2_lambda) / solv
            eq = [
                lh + g[h] + lb + g[hc] - ld - gd - water - constants[0],
                lh + g[h] + lc + g[co] - lb - g[hc] - constants[1],
                lh + g[h] + lo + g[oh] - water - constants[2],
            ]
            if coupled:
                eq.append(
                    lh + g[h] + g[so] - g[hs] - np.log(ratio) - constants[3]
                )
            return np.array(
                [
                    *eq,
                    np.log((b + c + d) / carbon),
                    (ion @ p.charge)
                    / (
                        max(ion @ abs(p.charge), 1e-300)
                        if extension
                        else charge_scale
                    ),
                ]
            )

        # Deterministic ideal-solution pH seed from charge and analytical totals.
        def seed(ph):
            proton = np.exp(-ph)
            ratio1, ratio2 = np.exp(constants[:2]) / proton
            d = carbon / (1 + ratio1 * (1 + ratio2))
            b = d * ratio1
            c = b * ratio2
            o = np.exp(constants[2]) * solvent0 / proton
            hydrogen = proton * solvent0
            hs_ratio = proton / np.exp(constants[3])
            net = (
                spectator
                + hydrogen
                - o
                - b
                - 2 * c
                - sulfur * (2 + hs_ratio) / (1 + hs_ratio)
            )
            return net, [b, c, d, o, hydrogen, *([hs_ratio] if coupled else [])]

        ph = brentq(lambda v: seed(v)[0], -50, 100)
        seed_amounts = np.asarray(seed(ph)[1])
        if not np.all(np.isfinite(seed_amounts) & (seed_amounts > 0)):
            raise ConvergenceError(
                "unrepresentable carbonate starting composition"
            )
        initial = np.log(seed_amounts)
        for retries, shift in enumerate((0.0, 0.25, -0.25, 1.0, -1.0)):
            start = initial.copy()
            start[:5] += shift * np.array([1, 1, -1, -1, 1])
            ans = root(
                residual,
                start,
                method="hybr",
                options=dict(
                    xtol=0.01 * np.sqrt(np.finfo(float).eps),
                    maxfev=CARBONATE_EVALUATIONS_PER_DIMENSION
                    * (len(start) + 1),
                    factor=100.0,
                    diag=np.ones(len(start)),
                ),
            )
            f = residual(ans.x)
            checks[0] = np.max(abs(f[:-2]))
            if checks[0] <= 1e-8 and max(abs(f[-2:])) <= 1e-10:
                final = [ion[hc], ion[co], neutral[p.co2], ion[oh], ion[h]]
                if (
                    extension
                    or np.all(final <= limits * (1 + 1e-10))
                    and (
                        not coupled
                        or ion[hs] <= min(proton_max, sulfur) * (1 + 1e-10)
                    )
                ):
                    break
        else:
            raise ConvergenceError(
                f"no carbonate root within inventory bounds; residual {f}"
            )
        checks[2] = abs((ion[hc] + ion[co] + neutral[p.co2]) - carbon) / carbon
    elif extension:
        # At the carbon-free face water still self-ionizes and, where present,
        # bisulfate re-equilibrates. Retain the exact O and H budgets.
        passive = i0.copy()
        passive[[oh, h]] = 0.0
        if sulfur:
            passive[[hs, so]] = 0.0
        spectator = passive @ p.charge

        def zero_carbon(v):
            nonlocal calls
            calls += 1
            proton, hydroxide = np.exp(np.clip(v[:2], -700, 50))
            ion[h], ion[oh] = proton, hydroxide
            neutral[p.water] = n0[p.water] + i0[oh] - hydroxide
            if sulfur:
                ratio = np.exp(np.clip(v[2], -700, 50))
                ion[hs], ion[so] = (
                    sulfur * ratio / (1 + ratio),
                    sulfur / (1 + ratio),
                )
            if neutral[p.water] <= 0:
                return np.full(len(v), 1e3 + abs(neutral[p.water]))
            g, water, solv = trial_logs(p, neutral, ion, t)
            lh = v[0] - np.log(solv) + g[h]
            eqs = [lh + v[1] - np.log(solv) + g[oh] - water - constants[2]]
            if sulfur:
                eqs.append(lh + g[so] - g[hs] - v[2] - constants[3])
            eqs.append((ion @ p.charge) / max(ion @ abs(p.charge), 1e-300))
            return np.asarray(eqs)

        def guess(logh):
            concentration = np.exp(logh)
            sulfate_ratio = concentration / np.exp(constants[3])
            net = (
                spectator
                + initial_solvent
                * (concentration - np.exp(constants[2]) / concentration)
                - sulfur * (2 + sulfate_ratio) / (1 + sulfate_ratio)
            )
            return net

        logh = brentq(guess, -100, 50)
        start = [
            logh + np.log(initial_solvent),
            constants[2] - logh + np.log(initial_solvent),
        ]
        if sulfur:
            start.append(logh - constants[3])
        answer = root(zero_carbon, start, tol=1e-11)
        checks[0] = max(abs(zero_carbon(answer.x)[:-1]))
    elif (
        coupled
        and min(i0[h] + i0[hs], sulfur)
        > 0.001 * np.finfo(float).eps * initial_solvent
    ):
        total_h = i0[h] + i0[hs]
        bound = min(total_h, sulfur)

        def residual(bisulfate):
            nonlocal calls
            calls += 1
            ion[hs], ion[so], ion[h] = (
                bisulfate,
                sulfur - bisulfate,
                total_h - bisulfate,
            )
            g, _, solv = trial_logs(p, neutral, ion, t)
            return (
                np.log(max(ion[h], 1e-300))
                + np.log(max(ion[so], 1e-300))
                - np.log(max(bisulfate, 1e-300))
                - np.log(solv)
                + g[h]
                + g[so]
                - g[hs]
                - constants[3]
            )

        try:
            b = brentq(
                residual,
                0.0,
                bound,
                xtol=np.nextafter(0.0, 1.0),
                maxiter=200,
                rtol=1e-15,
            )
        except (ValueError, RuntimeError):
            # Frozen-coefficient quadratic fallback from the sulfate mass-action law.
            scale = 0.001 * np.finfo(float).eps * initial_solvent
            wf = 0.097071 / (
                0.097071 + 2 * n0[p.water] * p.neutral_mw[p.water] / i0.sum()
            )
            alpha = (
                1
                - (1 + wf ** (-0.263468) - wf ** (-0.263204)) ** (-73.0039)
                + 1.72067 * wf**0.745718 * (1 - wf) ** 1.75
            )
            alpha = np.clip(alpha, 0.0, 1.0)
            guess = (
                alpha * scale if alpha > 0.4 else (1 - alpha) * (bound - scale)
            )
            residual(guess)
            g, _, solv = trial_logs(p, neutral, ion, t)
            linear = (
                total_h
                + sulfur
                + solv
                * np.exp(
                    constants[3] - np.clip(g[h] + g[so] - g[hs], -300.0, 300.0)
                )
            )
            b = (
                2
                * total_h
                * sulfur
                / (linear + np.sqrt(linear**2 - 4 * total_h * sulfur))
            )
            retries = 1
        checks[0] = abs(residual(b))
    if initial_sulfur:
        checks[3] = (
            abs(ion[hs] + ion[so] + solid - initial_sulfur) / initial_sulfur
        )
    checks[1] = abs(ion @ p.charge) / max(ion @ abs(p.charge), 1e-300)
    if (
        not np.all(np.isfinite(checks))
        or checks[0] > 1e-8
        or max(checks[1:]) > 1e-10
    ):
        raise ConvergenceError(f"equilibrium verification {checks}")
    return checks, calls, retries, solid
