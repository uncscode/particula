# %% Private charged-octyl-sulfate extension of the pinned Particula activity model.
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from particula_aiomfac.model import _prepare_fixed, _tables

HERE = Path(__file__).resolve().parent
SOS = 301  # study-local identifier outside the pinned database
MW_WATER = 0.01801528
MW_SOS = 0.232273
MW_NACL = 0.058443
MW_AS = 0.13214


def prepare(salt, tail, sodium=(0.0241, 0.224, 0.006081), ammonium_delta=0.0):
    """Ethyl sulfate homolog plus six covalently bound methylenes; fixed 1- charge."""
    comp = {"water": {16: 1}, "NaOS": {202: 1, 255: 1}}
    if salt == "NaCl":
        comp["NaCl"] = {202: 1, 242: 1}
    elif salt == "AS":
        comp.update(
            AS={204: 2, 261: 1}, Na2SO4={202: 2, 261: 1}, NH4OS={204: 1, 255: 1}
        )
    else:
        raise ValueError(salt)
    p = _prepare_fixed(comp)
    ids = tuple(SOS if x == 255 else x for x in p.ions)
    ia = p.ions.index(255) - p.ncation
    lane = p.ions.index(255) + 1
    t = _tables()
    counts = np.pad(p.counts, ((0, 0), (0, 1)))
    counts[lane, -1] = 6
    q = np.r_[p.q, t["SR_QQ"][1]]
    volume = p.volume.copy()
    volume[lane] += 6 * t["SR_RR"][1]
    inter = np.pad(p.interaction, ((0, 1), (0, 1)))
    inter[0, -1] = tail * t["ARR"][6, 0]
    inter[-1, 0] = tail * t["ARR"][0, 6]
    mass = p.salt_mw.copy()
    mass += p.stoich[:, p.ions.index(255)] * (
        MW_SOS - 0.02299 - t["SMWA"][255 - 241] * 0.001
    )
    b = tuple(x.copy() for x in p.bpair)
    c = tuple(x.copy() for x in p.cpair)
    for table, value in zip([b[0], b[1], c[1]], sodium):
        table[0, ia] = value
    if salt == "AS":
        b[0][1, ia] += ammonium_delta
    return replace(
        p,
        ions=ids,
        pairs=tuple((ca, SOS if an == 255 else an) for ca, an in p.pairs),
        estimated_pairs=(),
        salt_mw=mass,
        counts=counts,
        q=q,
        volume=volume,
        area=counts @ q,
        interaction=inter,
        bpair=b,
        cpair=c,
    )


def states(model, salt, water, na, os, temp):
    """Inventories and chemical potentials per mole of positive charge equivalents."""
    w, u, v = np.broadcast_arrays(water, na, os)
    w = w.ravel()
    u = u.ravel()
    v = v.ravel()
    if salt == "NaCl":
        basis = np.column_stack((w, v, 1 - v))
        counts = np.column_stack((w, np.ones_like(v), 1 - v, v))
    else:
        basis = np.column_stack(
            (w, u * v, (1 - u) * (1 - v) / 2, u * (1 - v) / 2, (1 - u) * v)
        )
        # model ion order came from donor IDs, even though octyl sulfate now has local ID 301
        cols = {202: u, 204: 1 - u, 261: (1 - v) / 2, SOS: v}
        counts = np.column_stack([w] + [cols[i] for i in model.ions])
    mass = basis * np.r_[MW_WATER, model.salt_mw]
    r = model.evaluate(mass, temp)
    water_mu = np.log(r.neutral_mole_fractions[:, 0]) + r.neutral_log_coefficients[:, 0]
    ions = np.log(np.maximum(r.ion_molalities, 1e-300)) + np.nan_to_num(
        r.ion_log_coefficients, nan=0.0
    )
    mu = np.column_stack((water_mu, ions))
    return counts, mu


def binary_prediction(tail, params, data):
    model = prepare("NaCl", tail, params)
    val = []
    for temp, g in data.groupby("T_K", sort=False):
        _, mu = states(model, "NaCl", 1 / (MW_WATER * g.m_SOS_mol_kg_water), 1, 1, temp)
        val.extend(zip(g.index, mu[:, 0]))
    return pd.Series(dict(val)).reindex(data.index).to_numpy()


def binary_fit(tail, weights=None):
    data = pd.read_csv(HERE / "binary-water.csv")
    # Resolution weights: reported meter accuracy; independent imposed RH gets a 1-point sensitivity scale.
    scales = (
        np.where(data.a_w_meter_accuracy.notna(), data.a_w_meter_accuracy, 0.01)
        / data.a_w.to_numpy()
    )
    if weights is not None:
        scales = scales * weights
    base = binary_prediction(tail, [0, 0, 0], data)
    design = np.stack(
        [binary_prediction(tail, v, data) - base for v in np.eye(3)], axis=1
    )
    bound = ([0.001, -3.0, -0.20], [0.6, 3.0, 0.2])
    sol = lsq_linear(
        design / scales[:, None],
        (np.log(data.a_w) - base) / scales,
        bounds=bound,
        tol=1e-13,
    )
    return (
        sol.x,
        np.exp(base + design @ sol.x),
        np.linalg.svd(design / scales[:, None], compute_uv=False),
    )


def feed_ratio(salt, ovf, rho=1.266):
    """Density in g/cm3. The SOS estimate is profiled separately from the fit."""
    organic = ovf / 100 * rho / MW_SOS
    equiv = (1 - ovf / 100) * (2.16 / MW_NACL if salt == "NaCl" else 2 * 1.77 / MW_AS)
    return organic / (organic + equiv)
