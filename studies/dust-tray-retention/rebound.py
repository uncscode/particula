"""Post-30-minute dry-dust delivery, bounce, re-impact and detachment."""

import csv
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from zipfile import ZipFile

import numpy as np
from defusedxml import ElementTree
from scipy.constants import elementary_charge, epsilon_0
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import t as student_t
from scipy.stats import ttest_ind
from study import read_workbook
from transport import (
    BOOK,
    C0_S1,
    DENSITY,
    HEIGHT,
    NU,
    ROOT,
    WINDS,
    initial,
    inputs,
    median,
    normalize,
    properties,
    save,
    tv,
    velocities,
)
from tribocharging import source

AREA = np.pi * 0.1**2
WORKBOOK_AREA = 3.14 * 0.254**2 / 4
HOURS = (0.5, 1.0, 1.5, 2.0, 3.0)


def table(name):
    """Read named rows from a saved study table."""
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))


@lru_cache
def trace(w):
    """Align the archived minute series to the low-wind onset."""
    tc, vc = (("H", "I"), ("K", "L"), ("N", "O"), ("Q", "R"))[w // 2]
    vals = []
    with ZipFile(BOOK) as book:
        for row in ElementTree.fromstring(
            book.read("xl/worksheets/sheet3.xml")
        ).findall(".//{*}row"):
            r = {
                re.sub(r"\d+", "", c.get("r")): float(c.findtext("{*}v"))
                for c in row
                if c.get("t") != "s" and c.findtext("{*}v") is not None
            }
            if tc in r and vc in r:
                vals.append([r[tc], r[vc]])
    t, c = np.array(vals).T
    start = np.flatnonzero(abs(c - C0_S1[w // 2]) < 0.0051)
    if not (len(start) == 1):
        raise ValueError("Expected a unique onset concentration")
    t = (t - t[start[0]]) * 60
    if not (np.all(np.diff(t) == 60) and np.all(c >= 0)):
        raise ValueError("Invalid monitor timestamps or concentration")
    return t, c


@lru_cache
def setup(w, n):
    """Construct the initial tray and layer inventories for this wind."""
    d, p, obs, u, _, c30 = inputs(w)
    x, m = initial(w, n, False)
    total = next(r for r in table("total-mass.csv") if int(r["wind_ms"]) == w)
    early = p * float(total["early_mg"])
    expected = float(total["measured_mg"])
    air = x[:n] * m[None, :]
    air = air * c30 / air.mean(axis=0).sum()
    v = velocities(d, DENSITY)
    _, _, _, diff = properties(d, 0)
    side = 4 / (np.pi * 2) * np.sqrt(u * u / NU * diff)
    return d, early, obs, expected, v, air, side, u


@dataclass(frozen=True)
class Case:
    """Surface response, geometry and impact parameters for one scenario."""

    name: str
    alpha: float = 0.0  # assumed normal fluctuation SD / friction velocity
    vc: float = np.inf  # capture threshold at 10 µm, m/s
    charge: float = 0.0  # grain-contact scale, µC/m²
    retry: float = 0.0  # probability that a bounce returns locally
    detachment: str = "none"
    rate: float = 0.0  # reference removal rate per hour
    area: float = AREA
    n: int = 12


def rates(w, case):
    """Return gross and captured normal speeds and the detachment rate."""
    d, _, _, _, v, _, _, u = setup(w, case.n)
    s = case.alpha * u
    threshold = np.full_like(d, np.inf)
    if np.isfinite(case.vc):
        # Effective neutral adhesion energy scales with projected area.
        vc2 = case.vc**2 * 10 / d
        if case.charge:
            _, m, _, _, z, _ = source(w, case.n, case.charge)
            # Point charge a radius from a flat dielectric. Account for the
            # gain in incoming energy as well as the work required to escape.
            beta = (3.0 - 1) / (3.0 + 1)
            image_work = (
                beta
                * (z * elementary_charge) ** 2
                / (16 * np.pi * epsilon_0 * d * 0.5e-6)
            )
            vc2 += 2 * image_work / m * (1 / 0.5**2 - 1)
        threshold = np.sqrt(vc2)
    if s == 0:
        gross = v
        capture = np.where(v <= threshold, v, 0.0)
    else:
        a = -v / s
        b = (threshold - v) / s
        phi_a = np.exp(-0.5 * a**2) / np.sqrt(2 * np.pi)
        phi_b = np.exp(-0.5 * b**2) / np.sqrt(2 * np.pi)
        gross = v * ndtr(-a) + s * phi_a
        capture = v * (ndtr(b) - ndtr(a)) + s * (phi_a - phi_b)
    if not (np.all(capture >= -1e-14) and np.all(capture <= gross + 1e-14)):
        raise ValueError("Capture speed outside physical bounds")
    capture = np.clip(
        capture, 0, gross
    )  # roundoff in the truncated normal integral
    loss = np.zeros_like(d)
    if case.detachment == "high_wind" and w == 6:
        loss = case.rate * (d / 10) / 3600
    elif case.detachment == "fine_stress":
        loss = case.rate * (d <= 10) / 3600
    return gross, capture, loss


def closed_run(w, case):
    """Advance the finite column and tray inventories from 30 minutes."""
    d, early, _, _, v, air, side, u = setup(w, case.n)
    gross, cap, loss = rates(w, case)
    n = case.n
    dz = HEIGHT / n
    floor, wall, out, hit, bounced, released = range(n, n + 6)
    gen = np.zeros((len(d), n + 6, n + 6))
    z = np.arange(1, n) * dz
    for j, k in enumerate(0.4 * u * z * (1 - z / HEIGHT) / dz**2):
        gen[:, j + 1, j] += k
        gen[:, j, j + 1] += k
    for j in range(n):
        if j:
            gen[:, j - 1, j] += v / dz
        else:
            gen[:, floor, j] += cap / dz
            gen[:, out, j] += (gross - cap) * (1 - case.retry) / dz
        gen[:, wall, j] += side
        gen[:, j, j] = -gen[:, : out + 1, j].sum(axis=1)
    gen[:, floor, floor] = -loss
    gen[:, out, floor] = loss
    gen[:, hit, 0] = gross / dz
    gen[:, bounced, 0] = (gross - cap) / dz
    gen[:, released, floor] = loss
    x = np.zeros((n + 6, len(d)))
    x[:n] = case.area * dz * air  # mg per parcel over the tray footprint
    x[floor] = early
    total = x[: out + 1].sum(axis=0)
    outputs = []
    for t in HOURS:
        y = np.einsum(
            "sij,js->is", expm(gen * ((t - 0.5) * 3600)), x, optimize=False
        )
        balance = np.max(np.abs(y[: out + 1].sum(axis=0) - total)) / total.sum()
        if not (y.min() > -1e-11 and balance < 1e-9):
            raise ValueError("Closed mass budget failed")
        outputs.append((t, y[floor], y[hit], y[bounced], y[released], balance))
    return outputs


@lru_cache
def forcing(w, t):
    """Integrate the local proxy for airborne exposure at each retrieval."""
    ts, c = trace(w)
    stop = t * 3600
    edges = np.r_[1800.0, ts[(ts > 1800) & (ts < stop)], stop]
    abscissa, weight = np.polynomial.legendre.leggauss(3)
    widths = np.diff(edges)
    points = (
        (edges[:-1] + edges[1:])[:, None] / 2 + widths[:, None] * abscissa / 2
    ).ravel()
    weights = (widths[:, None] * weight / 2).ravel()
    # Constant final observed value fills the unrecorded final 4–7 min.
    return stop - points, np.interp(points, ts, c) * weights


def open_run(w, case, shape):
    """Integrate deposition and detachment for a replenished air supply."""
    d, early, _, _, v, air, _, _ = setup(w, case.n)
    if shape == "depleted":
        f = normalize(air.sum(axis=0))
    elif shape == "flux_matched":
        f = normalize(early / v)
    else:
        f = normalize(early)
    gross, cap, loss = rates(w, case)
    # Each returned impact independently resamples normal speed. Immediate
    # retries give an upper sensitivity to local recollection in open flow.
    retry = case.retry * (1 - cap / gross)
    arrivals = case.area * f * gross / (1 - retry)
    captured = case.area * f * cap / (1 - retry)
    outputs = []
    for t in HOURS:
        age, weights = forcing(w, t)
        exposure = sum(weights)
        surviving_exposure = weights @ np.exp(-age[:, None] * loss)
        retained = (
            early * np.exp(-loss * (t - 0.5) * 3600)
            + captured * surviving_exposure
        )
        hit = arrivals * exposure
        bounced = (arrivals - captured) * exposure
        released = early * (-np.expm1(-loss * (t - 0.5) * 3600)) + captured * (
            weights @ (-np.expm1(-age[:, None] * loss))
        )
        error = max(abs(retained - early - hit + bounced + released)) / (
            early.sum() + hit.sum()
        )
        if not (error < 1e-10 and np.all(retained >= 0)):
            raise ValueError("Open mass budget failed")
        outputs.append((t, retained, hit, bounced, released, error))
    return outputs


def cutoff_band(d, delta):
    """Interpolate cumulative mass change at both exact diameter cutoffs."""
    cumulative = np.interp(np.log([10, 20]), np.log(d), np.cumsum(delta))
    return cumulative[1] - cumulative[0]


def diagnostics(w, case, supply, out):
    """Summarize the retained spectrum and verify its mass budget."""
    d, early, obs, expected, _, _, _, _ = setup(w, case.n)
    middle = (d > 10) & (d <= 20)
    result = []
    for t, ret, hit, bounce, release, error in out:
        result.append(
            dict(
                wind_ms=w,
                supply=supply,
                case=case.name,
                hours=t,
                area_m2=case.area,
                layers=case.n,
                normal_sd_over_ustar=case.alpha,
                vc10_ms=case.vc,
                charge_scale_uC_m2=case.charge,
                return_probability=case.retry,
                removal_model=case.detachment,
                rate10_per_h=case.rate,
                retained_mg=sum(ret),
                net_gain_mg=sum(ret - early),
                middle_gain_mg=sum((ret - early)[middle]),
                cutoff_middle_gain_mg=cutoff_band(d, ret - early),
                attempted_mg=sum(hit),
                bounced_mg=sum(bounce),
                detached_mg=sum(release),
                median_um=median(d, ret),
                early_median_um=median(d, early),
                observed_3h_median_um=median(d, obs),
                observed_3h_mg=expected,
                tv_to_3h_pp=tv(normalize(ret), normalize(obs)),
                balance_error=error,
            )
        )
    return result


def bounds(w):
    """Bound the supply required by the observed mass in each band."""
    ts, c = trace(w)
    pre = (ts >= 0) & (ts <= 1800)
    early_exposure = np.trapezoid(c[pre], ts[pre])
    _, weight = forcing(w, 3)
    exposure = sum(weight)
    early, late = (setup(w, 12)[1].sum(), setup(w, 12)[3])
    dt_tail = 10800 - ts[-1]
    masses = [
        r for r in table("threshold-masses.csv") if int(r["wind_ms"]) == w
    ]
    rel = []
    for area in (AREA, WORKBOOK_AREA):
        for trace_case, integral in [
            ("last", exposure),
            ("zero_tail", exposure - dt_tail * c[-1]),
            ("max_late_tail", exposure + dt_tail * (max(c[-10:]) - c[-1])),
        ]:
            for r in masses:
                hi = float(r["upper_um"])
                gain = float(r["increment_mg"])
                vmax = float(velocities(hi, DENSITY))
                rel.append(
                    dict(
                        wind_ms=w,
                        area_m2=area,
                        tail=trace_case,
                        upper_um=hi,
                        early_exposure_mg_s_m3=early_exposure,
                        late_exposure_mg_s_m3=integral,
                        fill_minutes=dt_tail / 60,
                        measured_gain_mg=gain,
                        max_settling_ms=vmax,
                        min_airborne_mass_fraction=max(0, gain)
                        / (area * integral * vmax),
                        required_total_net_speed_ms=(late - early)
                        / (area * integral),
                        empirical_3h_mg=early * (1 + integral / early_exposure),
                        closed_remaining_mg=area * HEIGHT * inputs(w)[-1],
                    )
                )
    return rel


def changing_surface():
    """Condition on the early tray and change capture at 30 minutes."""
    w = 2
    d, early, obs, _, _, _, _, _ = setup(w, 12)
    ref = bounds(w)[0]
    ie, il = ref["early_exposure_mg_s_m3"], ref["late_exposure_mg_s_m3"]
    keep = early > 0
    middle = (d > 10) & (d <= 20)
    rows = []
    for alpha in (0.25, 1.0, 2.0):
        for first in (0.01, 0.05, 0.2):
            case = Case("early", alpha=alpha, vc=first)
            _, k0, _ = rates(w, case)
            if not (np.all(k0[keep] > 0)):
                raise ValueError("Early capture inversion is singular")
            inv = early[keep] / (AREA * ie * k0[keep])
            scale = inv.sum()
            air_shape = normalize(inv)
            for last in (first, 0.05, 0.2, np.inf):
                if last < first or any(
                    r["alpha"] == alpha
                    and r["early_vc10_ms"] == first
                    and r["late_vc10_ms"] == last
                    and r["late_charge_scale"] == 0
                    for r in rows
                ):
                    continue
                for charge in (0.0, 10.0) if first == last else (0.0,):
                    _, k1, _ = rates(w, replace(case, vc=last, charge=charge))
                    delta = np.zeros(len(d))
                    delta[keep] = AREA * il * scale * air_shape * k1[keep]
                    if first == last and charge == 0:
                        if not (np.allclose(delta, early * il / ie)):
                            raise ValueError(
                                "Fixed-surface exposure identity failed"
                            )
                    rows.append(
                        dict(
                            alpha=alpha,
                            early_vc10_ms=first,
                            late_vc10_ms=last,
                            late_charge_scale=charge,
                            local_bulk_multiplier=scale,
                            net_gain_mg=delta.sum(),
                            middle_gain_mg=delta[middle].sum(),
                            cutoff_middle_gain_mg=cutoff_band(d, delta),
                            median_um=median(d, early + delta),
                            tv_pp=tv(normalize(early + delta), normalize(obs)),
                        )
                    )
    return rows


def maximum_delivery():
    """Find the minimum incident dispersion for ideal capture at 2 m/s."""
    rows = []
    w = 2
    bs = [x for x in bounds(w) if x["area_m2"] == AREA and x["tail"] == "last"]
    g = np.array([x["measured_gain_mg"] for x in bs])
    v = np.array([x["max_settling_ms"] for x in bs])
    ei, li = bs[0]["early_exposure_mg_s_m3"], bs[0]["late_exposure_mg_s_m3"]
    early = np.array(
        [
            float(r["early_mg"])
            for r in table("threshold-masses.csv")
            if int(r["wind_ms"]) == 2
        ]
    )
    u = inputs(w)[3]
    for area in (AREA, WORKBOOK_AREA):

        def demand(a, area=area):
            sd = a * u
            k = (
                v
                if a == 0
                else v * ndtr(v / sd)
                + sd * np.exp(-0.5 * (v / sd) ** 2) / np.sqrt(2 * np.pi)
            )
            late = np.maximum(0, g) / (area * li * k)
            return late.sum(), np.maximum(early / (area * ei * k), late).sum()

        for a in (
            0.0,
            0.25,
            0.5,
            1.0,
            2.0,
            3.0,
            brentq(lambda a: demand(a)[1] - 1, 0.0, 5.0),
        ):
            late, both = demand(a)
            rows.append(
                dict(
                    area_m2=area,
                    normal_sd_over_ustar=a,
                    sd_ms=a * u,
                    late_min_fraction_sum=late,
                    both_windows_min_fraction_sum=both,
                )
            )
    return rows


def timeline():
    """Compare retrieval group means with integrated monitor exposure."""
    rows = []
    kinetic = read_workbook()[2]
    for w in WINDS:
        archive = {
            (r["E"], r["D"]): 1000 * r["F"]
            for r in kinetic.values()
            if r.get("C") == w and "F" in r
        }
        for band in ("all", "middle"):
            masses = [
                archive[t, 63]
                if band == "all"
                else archive[t, 20] - archive[t, 10]
                for t in HOURS
            ]
            for t, measured in zip(HOURS, masses, strict=True):
                fraction = sum(forcing(w, t)[1]) / sum(forcing(w, 3.0)[1])
                observed_frac = (measured - masses[0]) / (
                    masses[-1] - masses[0]
                )
                rows.append(
                    dict(
                        wind_ms=w,
                        band=band,
                        time_h=t,
                        observed_mg=measured,
                        observed_fraction_of_increment=observed_frac,
                        exposure_fraction=fraction,
                        endpoint_constrained_mg=masses[0]
                        + fraction * (masses[-1] - masses[0]),
                    )
                )
    return rows


def timing_contrasts(history):
    """Two m/s PM63 retrieval differences, conditional on measured exposure."""
    totals = {
        r["time_h"]: r
        for r in history
        if r["wind_ms"] == 2 and r["band"] == "all"
    }
    kinetic = read_workbook()[2]
    sem = {
        r["E"]: r["H"] * WORKBOOK_AREA
        for r in kinetic.values()
        if r.get("C") == 2 and r.get("D") == 63
    }
    baseline = next(r for r in table("total-mass.csv") if r["wind_ms"] == "2")
    for t, col in [(0.5, "early_sem_mg"), (3.0, "measured_sem_mg")]:
        if not (abs(sem[t] - float(baseline[col])) < 0.001):
            raise ValueError("Archived endpoint standard errors disagree")
    rows = []
    for t in HOURS[1:-1]:
        r = totals[t]
        f = r["exposure_fraction"]
        gap = r["endpoint_constrained_mg"] - r["observed_mg"]
        variances = np.square([(1 - f) * sem[0.5], sem[t], f * sem[3.0]])
        # Five different trays per time. Fix the measured exposure fraction.
        variance = variances.sum()
        df = variance**2 / (variances**2 / 4).sum()
        delta = student_t.ppf(0.975, df) * np.sqrt(variance)
        rows.append(
            dict(
                time_h=t,
                exposure_fraction=f,
                expected_minus_observed_mg=gap,
                group_sem_mg=sem[t],
                contrast_se_mg=np.sqrt(variance),
                welch_df=df,
                low95_mg=gap - delta,
                high95_mg=gap + delta,
            )
        )
    return rows


def gradual_surface():
    """Approach the prescribed late capture condition exponentially."""
    rows = []
    w = 2
    d, early, obs, _, _, _, _, _ = setup(w, 12)
    middle = (d > 10) & (d <= 20)
    ie = bounds(w)[0]["early_exposure_mg_s_m3"]
    keep = early > 0
    for a in (1.0, 2.0):
        ref = Case("early", alpha=a, vc=0.05)
        _, k0, _ = rates(w, ref)
        total_scale = sum(early[keep] / (AREA * ie * k0[keep]))
        for mechanism in ("mechanical", "image"):
            for tau in (15.0, 60.0, 150.0):
                for t in HOURS:
                    age, weights = forcing(w, t)
                    after = (t - 0.5) * 60 - age / 60
                    progress = -np.expm1(-after / tau)
                    v = setup(w, 12)[4]
                    sd = a * inputs(w)[3]
                    if mechanism == "mechanical":
                        vc = (0.05 + 0.15 * progress[:, None]) * np.sqrt(10 / d)
                    else:
                        _, m, _, _, z, _ = source(w, 12, 10.0)
                        work = (
                            0.5
                            * (z * elementary_charge) ** 2
                            / (16 * np.pi * epsilon_0 * d * 0.5e-6)
                        )
                        vc = np.sqrt(
                            0.05**2 * 10 / d
                            + progress[:, None] ** 2 * 6 * work / m
                        )
                    lo = -v / sd
                    hi = (vc - v) / sd
                    cap = v * (ndtr(hi) - ndtr(lo)) + sd / np.sqrt(
                        2 * np.pi
                    ) * (np.exp(-0.5 * lo**2) - np.exp(-0.5 * hi**2))
                    delta = np.zeros(len(d))
                    delta[keep] = (
                        early[keep] / ie * (weights @ cap[:, keep]) / k0[keep]
                    )
                    rows.append(
                        dict(
                            alpha=a,
                            change=mechanism,
                            tau_min=tau,
                            hours=t,
                            local_bulk_multiplier=total_scale,
                            net_gain_mg=delta.sum(),
                            middle_gain_mg=sum(delta[middle]),
                            cutoff_middle_gain_mg=cutoff_band(d, delta),
                            median_um=median(d, early + delta),
                        )
                    )
    return rows


def interval_checks():
    """Compute independent-group intervals for measured cumulative gains."""
    rows = []
    masses = read_workbook()[1]
    for area in (AREA, WORKBOOK_AREA):
        for b in bounds(2):
            if b["area_m2"] != area or b["tail"] != "last":
                continue
            c = b["upper_um"]
            pre, post = [masses[2, t, c] for t in (0.5, 3.0)]
            ci = ttest_ind(post, pre, equal_var=False).confidence_interval()
            rows.append(
                dict(
                    cut_um=c,
                    area_m2=area,
                    cumulative_gain_mg=np.mean(post) - np.mean(pre),
                    welch_low_mg=ci.low,
                    welch_high_mg=ci.high,
                    perfect_gravity_ceiling_mg=area
                    * b["late_exposure_mg_s_m3"]
                    * b["max_settling_ms"],
                )
            )
    return rows


def surface_cases(w):
    """Enumerate surface scenarios for the specified wind."""
    cases = [
        Case("perfect_gravity"),
        Case("perfect_quarter", alpha=0.25),
        Case("perfect_turbulent", alpha=1),
    ]
    for a in (0.0, 0.25, 1.0):
        for vc in (0.01, 0.05, 0.2):
            cases.append(Case(f"bounce_{a}_{vc}", alpha=a, vc=vc))
    ref = Case("bounce_ref", alpha=1, vc=0.05)
    cases.extend(
        [
            replace(ref, name="half_return", retry=0.5),
            replace(ref, name="high_return", retry=0.9),
            replace(ref, name="image_1", charge=1),
            replace(ref, name="image_10", charge=10),
            replace(ref, name="alternate_area", area=WORKBOOK_AREA),
            Case("alternate_perfect_turbulent", alpha=1, area=WORKBOOK_AREA),
            replace(ref, name="refined_layers", n=24),
        ]
    )
    if w == 6:
        for rate in (0.1, 1.0):
            cases.append(
                replace(
                    ref,
                    name=f"mobilize_{rate}",
                    detachment="high_wind",
                    rate=rate,
                )
            )
    if w == 2:
        cases.append(
            replace(
                ref,
                name="selective_fine_removal",
                detachment="fine_stress",
                rate=1.0,
            )
        )
    return cases


def fitted_removal(w):
    """Fit late fine-particle removal to the observed three-hour median."""
    base = Case(
        "fitted_fine_removal", alpha=1, vc=0.05, detachment="fine_stress"
    )
    d, _, obs, *_ = setup(w, 12)
    target = median(d, obs)

    def error(k):
        output = open_run(w, replace(base, rate=k), "tray_shape")
        return median(d, output[-1][1]) - target

    fitted = replace(base, rate=brentq(error, 0, 1))
    return diagnostics(
        w, fitted, "tray_shape", open_run(w, fitted, "tray_shape")
    )


def main():
    """Run the four wind treatments and write the numerical study outputs."""
    results = []
    summaries = []
    channels = []
    for w in WINDS:
        for case in surface_cases(w):
            for supply in ("closed", "depleted", "flux_matched", "tray_shape"):
                out = (
                    closed_run(w, case)
                    if supply == "closed"
                    else open_run(w, case, supply)
                )
                results.extend(diagnostics(w, case, supply, out))
                if case.name in (
                    "perfect_turbulent",
                    "bounce_1.0_0.05",
                    "image_10",
                ):
                    d, early, obs, *_ = setup(w, case.n)
                    for size, start, measured_fraction, retained in zip(
                        d, early, obs, out[-1][1], strict=True
                    ):
                        channels.append(
                            dict(
                                wind_ms=w,
                                supply=supply,
                                case=case.name,
                                diameter_um=size,
                                early_tray_mg=start,
                                measured_3h_fraction=measured_fraction,
                                retained_3h_mg=retained,
                            )
                        )
        if w == 2:
            results.extend(fitted_removal(w))
        summaries.extend(bounds(w))
    save("rebound-channels.csv", channels)
    save("rebound-results.csv", results)
    save("rebound-bounds.csv", summaries)
    save("rebound-changing-surface.csv", changing_surface())
    save("rebound-delivery-limit.csv", maximum_delivery())
    save("rebound-mass-intervals.csv", interval_checks())
    history = timeline()
    save("rebound-timing.csv", history)
    save("rebound-timing-intervals.csv", timing_contrasts(history))
    save("rebound-gradual.csv", gradual_surface())


if __name__ == "__main__":
    main()
