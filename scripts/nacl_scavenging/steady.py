"""Apply Cai’s steady count retrieval to the simulated DMA response."""

from functools import lru_cache

import numpy as np
from instrument import (
    DILUTION,
    LOG_D,
    PRESSURE,
    STATES,
    TARGETS,
    TEMP,
    D,
    aliases,
    archive_table,
    attachment_coefficient,
    fit_walls,
    ion_density,
    probabilities,
    response,
    save,
    selected,
    vdw_table,
    wall_table,
)
from particula.dynamics import get_hard_sphere_kernel_via_system_state
from scipy.interpolate import RegularGridInterpolator


@lru_cache(maxsize=8)
def native_tables(stride):
    """Interpolate native collision rates by diameter and charge product."""
    # Positive charges on the first set and signed z1*z2 on the second set
    # give the exact Coulomb pair product. The returned cross block supplies
    # size interpolation independently of the chamber Monte Carlo sampler.
    small = D[::stride]
    small = small[(small >= 1.6) & (small <= 38)]
    large = D[::stride]
    large = large[(large >= 50) & (large <= 420)]
    radius = np.r_[small, large] * 0.5e-9
    m = (4 * np.pi / 3 * 2160) * radius**3
    tables = {}
    for product in sorted({i * j for i in range(-3, 4) for j in STATES}):
        charge = np.r_[np.ones(len(small)), np.full(len(large), product)]
        k = get_hard_sphere_kernel_via_system_state(
            radius, m, charge, TEMP, PRESSURE
        )
        tables[product] = RegularGridInterpolator(
            (np.log(small), np.log(large)),
            np.log(k[: len(small), len(small) :] * 1e6),
        )
    return tables


def split(joint, p, states, tube, parts):
    """Use equal-number quadrature within each transmitted charge branch."""
    inlet = p[:, states + 8].copy()
    inlet[:, states != 0] *= tube
    survival = inlet.sum(axis=1)
    inlet /= survival[:, None]
    joint = joint * survival[:, None]
    joint /= joint.sum()
    indices, weights = [], []
    for column in joint.T:
        if column.sum() < 1e-12:
            continue
        ids = np.searchsorted(
            np.cumsum(column), (np.arange(parts) + 0.5) * column.sum() / parts
        )
        indices.extend(ids)
        weights.extend(np.full(parts, column.sum() / parts))
    ix, inverse = np.unique(indices, return_inverse=True)
    weights = np.bincount(inverse, weights=weights)
    return ix, weights / weights.sum(), inlet[ix], joint.sum(axis=0)


def solve(ix, states, ions, bound, source, losses):
    """Solve stationary charge populations under injection and removal."""
    r = D[ix, None] * 0.5e-9
    neg = (
        1e6
        * ions[0]
        * attachment_coefficient(r, states[None, :], -1, omitted=bound)
    )
    pos = (
        1e6
        * ions[1]
        * attachment_coefficient(r, states[None, :], 1, omitted=bound)
    )
    neg[:, 0] = 0
    pos[:, -1] = 0
    a = np.zeros((len(ix), len(states), len(states)))
    k = np.arange(len(states))
    a[:, k, k] = losses + neg + pos
    a[:, k[:-1], k[1:]] = -neg[:, 1:]
    a[:, k[1:], k[:-1]] = -pos[:, :-1]
    steady = np.linalg.solve(a, source[:, :, None])[:, :, 0]
    np.testing.assert_equal(steady.min() > -1e-12, True)
    np.testing.assert_allclose((steady * losses).sum(), source.sum(), rtol=1e-9)
    return steady


@lru_cache(maxsize=2)
def probs(bound):
    """Cache the normalized charge distributions for a scenario."""
    return probabilities(D, bound)


def evaluate(
    d,
    case,
    *,
    parts=64,
    stride=8,
    bound="upper",
    parent=1.0,
    gsd=1.6,
    collector_width=0.1,
    analyzer=0.1,
    channel_centers=None,
    smode=1.0,
    sign=1,
    tube=0.1,
    ion=2.0,
    ideal=False,
    max_z=3,
    blur=0.03,
    cutoff=0.0,
    neutral=False,
    atomizer_multi=1.0,
):
    """Retrieve steady coefficients for one tracer channel and scenario."""
    p = probs(bound)
    atom = p.copy()
    atom[:, abs(STATES) >= 2] *= atomizer_multi
    atom /= atom.sum(axis=1, keepdims=True)
    ls = selected(
        100,
        atom,
        large=True,
        ideal=ideal,
        mode_ratio=parent,
        gsd=gsd,
        width=collector_width,
        sign=sign,
        max_z=max_z,
    )
    ss = selected(
        d,
        p,
        ideal=ideal,
        mode_ratio=smode,
        gsd=gsd,
        sign=sign,
        max_z=max_z,
        blur=blur,
    )
    small_z = np.arange(-3, 4)
    si, sw, sp, smix = split(ss, p, small_z, tube, parts)
    li, lw, lp, lmix = split(ls, p, STATES, 1.0, parts)
    wall = wall_table(fit_walls())
    centers = (d,) if channel_centers is None else channel_centers
    scan = sum(
        response(center, analyzer, blur, p, sign, max_z)[si]
        for center in centers
    )
    eff = (
        np.ones(len(si))
        if cutoff == 0
        else 1 / (1 + np.exp(-(D[si] - cutoff) / 0.65))
    )
    scan = scan * eff[:, None]
    readout = scan.sum(axis=1)
    ions = ion_density(p, ls.sum(axis=1), 1e4, ion)
    lloss = wall[li, None] + DILUTION
    large = solve(li, STATES, ions, bound, lw[:, None] * lp, lloss)
    large /= large.sum()
    # Fixed supply rate and steady concentration. Above 100 nm wall loss is
    # held at its measured 100 nm value as in the GPU pulses.
    small_log_d, large_log_d = np.meshgrid(
        np.log(D[si]), np.log(D[li]), indexing="ij"
    )
    query = np.stack([small_log_d, large_log_d], axis=-1)
    native = native_tables(stride)
    products = {a * b for a in small_z for b in STATES}
    kernels = {k: np.exp(native[k](query)) for k in products}
    kernel = np.column_stack(
        [
            sum(
                kernels[0 if neutral else a * b] @ large[:, j]
                for j, b in enumerate(STATES)
            )
            for a in small_z
        ]
    )
    extra = np.exp(dispersion_interpolator()(query)) @ large.sum(axis=1)
    # Published attraction curve at 100 nm anchors the absolute rate. Extend
    # only the ratio outside the archive, with its nearest endpoint value.
    archive = archive_table("Fig6_CoagCoef")
    ref = np.column_stack(
        (np.log(archive[:, 6]), np.full(len(archive), np.log(100.0)))
    )
    centered = np.exp(native[0](ref)) + np.exp(dispersion_interpolator()(ref))
    anchor = np.interp(np.log(D[si]), ref[:, 0], archive[:, 9] / centered)
    source = sw[:, None] * sp
    background = wall[si, None] + DILUTION
    blank = solve(
        si,
        small_z,
        ion_density(p, ls.sum(axis=1), 0, ion),
        bound,
        source,
        background,
    )
    baseline_signal = readout @ blank.sum(axis=1)
    loss0 = (readout @ sw) / baseline_signal
    results = []
    for name, k in [
        ("native", kernel),
        ("dispersion", kernel + extra[:, None]),
        ("published_anchor", (kernel + extra[:, None]) * anchor[:, None]),
    ]:
        state = solve(si, small_z, ions, bound, source, background + k * 1e4)
        sensor = scan.T @ state.sum(axis=1)
        apparent = (baseline_signal / sensor.sum() - 1) * loss0 / 1e4
        results.append(
            dict(
                case=case,
                model=name,
                target_nm=d,
                apparent_cm3_s=apparent,
                true_capture_cm3_s=(k * state).sum() / state.sum(),
                blank_loss_h=loss0 * 3600,
                normalized_count=sensor.sum() / baseline_signal,
                small_selected_double=smix[1] if len(smix) > 1 else 0,
                small_selected_triple=smix[2] if len(smix) > 2 else 0,
                large_selected_double=lmix[1] if len(lmix) > 1 else 0,
                large_selected_triple=lmix[2] if len(lmix) > 2 else 0,
                large_selected_higher=lmix[3:].sum(),
                sensor_multiple=sensor[1:].sum() / sensor.sum(),
                small_negative=state[:, small_z < 0].sum() / state.sum(),
                small_positive=state[:, small_z > 0].sum() / state.sum(),
                large_negative=large[:, STATES < 0].sum(),
                large_positive=large[:, STATES > 0].sum(),
                small_charged_inlet=1 - (source[:, small_z == 0]).sum(),
                large_charged_inlet=1 - (lw @ lp)[8],
                **{
                    f"small_z{z:+d}": state[:, small_z == z].sum() / state.sum()
                    for z in small_z
                },
                **{
                    f"large_z{z:+d}": large[:, STATES == z].sum()
                    for z in STATES
                },
                small_alias2_nm=aliases(d)[1],
                small_alias3_nm=aliases(d)[2],
            )
        )
    return results


@lru_cache(maxsize=1)
def dispersion_interpolator():
    """Cache the dispersion rate increment over log pair diameters."""
    return RegularGridInterpolator((LOG_D[::8], LOG_D[::8]), vdw_table())


if __name__ == "__main__":
    cases = [
        ("reference", {}),
        ("numerical", dict(parts=128, stride=4)),
        ("single_size", dict(ideal=True)),
        ("only_single_selection", dict(max_z=1)),
        ("through_six", dict(max_z=6)),
        ("no_coulomb", dict(neutral=True)),
        ("multi_depleted", dict(atomizer_multi=0.5)),
        ("multi_enriched", dict(atomizer_multi=2.0)),
        ("low_parent", dict(parent=0.7)),
        ("large_parent", dict(parent=1.6)),
        ("narrow_sources", dict(gsd=1.3)),
        ("broad_sources", dict(gsd=1.9)),
        ("larger_furnace_mode", dict(smode=1.6)),
        ("negative_dma", dict(sign=-1)),
        ("complete_tube_loss", dict(tube=0.0)),
        ("full_tube_transmission", dict(tube=1.0)),
        ("elevated", dict(ion=100.0)),
        ("low_attachment", dict(bound="lower")),
        ("wide_collector_dma", dict(collector_width=0.25)),
        ("wide_smps", dict(analyzer=0.25)),
        ("extra_diffusive_blur", dict(blur=0.07)),
        ("cpc_sensitivity", dict(cutoff=3.5)),
    ]
    rows = []
    for name, params in cases:
        for d in TARGETS:
            rows.extend(evaluate(d, name, **params))
    save("steady.csv", rows)
    # July 29 export in Cai, Zenodo 13291658, metadata/timeseries.
    # Sum seven log-spaced channels with equal dwell and a flat CPC response.
    recorded_scan = evaluate(
        4.0,
        "archived_4nm_scan",
        analyzer=1.5 / 5,
        channel_centers=(3.59, 3.72, 3.85, 4.00, 4.14, 4.29, 4.45),
    )
    save("archived-4nm-scan.csv", recorded_scan)
