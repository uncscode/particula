"""Reproduce reference-state rate coefficients and conditional kinetic hazards.

Run this file from the repository root. The 1000 K wet bath is a diagnostic
snapshot. The 1273 K dry case uses an experimental inlet. Rates use the
unrounded fit vectors; the reactor trajectories determine actual fluxes.
"""

import csv
import json
from math import exp, sqrt
from pathlib import Path

import numpy as np
import warp as wp
from forward import dry
from forward import kernels as k
from forward import wet_equilibrium as eq
from forward.common import F, Vec
from forward.gas import monomer_fraction
from forward.wet_network import Flux, Net, Pool
from forward.wet_stiff import stiff_gas
from scipy.constants import R

HERE = Path(__file__).resolve().parent
OUT = (
    HERE.parents[1]
    / "website/public/study-data/straw-combustion/reaction-rates.csv"
)
WET = "https://doi.org/10.1016/j.fuel.2024.130974"
DRY = "https://doi.org/10.1021/jp0557215"
BATH = (
    "5% O2, 5% H2O; KCl=200, SO2=50, SO3=1, K2SO4(g)=1 ppm; "
    "KOH=KHSO3=KHSO4=KClSO3=1 ppm; HCl=5 ppm by Cl balance; no aerosol"
)
FITS = ("split_combined", "all_combined")
Probe = wp.types.vector(10, F)


@wp.kernel(enable_backward=False)
def diagnostic(
    p: wp.array(dtype=k.Params),
    a: wp.array(dtype=Vec),
    b: wp.array(dtype=Net),
    dt: F,
    audit: wp.array(dtype=Flux),
    thermo: wp.array(dtype=Probe),
):
    """Probe the actual wet half-step with finite populations and recover Kp."""
    i = wp.tid()
    y = k.State()
    y[0] = F(50)
    y[1] = F(1)
    y[2] = F(200)
    y[3] = F(1)
    q = Pool(F(1), F(1), F(1), F(1))
    t = F(1000)
    one = F(1)
    zero = F(0)
    _state, _pool, flux = stiff_gas(
        y, q, p[i], a[i], b[i], k.Feed(F(200), F(50), F(5)), t, dt, 40, 0, zero
    )
    audit[i] = flux / dt
    thermo[i] = Probe(
        eq.kh(one, zero),
        eq.k11(one, zero),
        eq.k12(one, zero),
        eq.k13(one, zero),
        eq.k17b(one, zero),
        eq.k16(one, zero),
        eq.k19(one, zero),
        eq.k18(one, zero),
        eq.k20(one, zero),
        monomer_fraction(t, F(200)),
    )


def probe(x, dt):
    """Return native fluxes per second and thermochemical constants."""

    def arr(z, dtype):
        return wp.array(np.ascontiguousarray(z), dtype=dtype, device="cpu")

    flux = wp.empty(2, dtype=Flux, device="cpu")
    thermo = wp.empty(2, dtype=Probe, device="cpu")
    wp.launch(
        diagnostic,
        2,
        [
            arr(x[:, :7], k.Params),
            arr(x[:, 9:29], Vec),
            arr(x[:, 29:], Net),
            dt,
            flux,
            thermo,
        ],
        device="cpu",
    )
    return flux.numpy(), thermo.numpy()[0]


def record(
    reaction,
    quantity,
    source_value,
    source_unit,
    values,
    unit,
    condition,
    note,
    source=WET,
    t=1000.0,
    pressure=101325.0,
):
    """Put source and reduced-model values in separately labelled columns."""
    return dict(
        reaction=reaction,
        model_quantity=quantity,
        temperature_K=t,
        pressure_Pa=pressure,
        bath=condition,
        reference_coefficient=source_value,
        reference_units=source_unit,
        reference_origin="",
        reference_hazard_at_bath_s1="",
        split_fit=values[0],
        all_data_fit=values[1],
        model_units=unit,
        interpretation=note,
        source=source,
    )


def wet_rows(x):
    """Evaluate formulas and verify gross hazards against the wet kernel."""
    flux, constants = probe(x, 1e-13)
    kh, k11p, k12p, k13p, k17p, _, k19p, k18p, k20p, mono = constants
    c = 101325 / (R * 1000)
    one_ppm = c * 1e-6
    b = x[:, 29:].T

    def gate(n):
        return b[n - 1]

    stab_b, stab_a, stab_c = np.exp(12552 / (R * 1000) * b[35:38])

    def modifier(n):
        return np.exp(6000 / (R * 1000) * gate(n))

    def falloff(a, power, scale):
        low = a * 1000**power * c * 10.0**scale
        return low / (1 + low / 1e8)

    k11 = falloff(2.6e30, -7.6, gate(33))
    k12 = falloff(2.6e30, -7.6, gate(34))
    k0 = falloff(2.6e30, -7.6, 0)
    k13 = falloff(1.9e29, -7.8, gate(35))
    k13_base = falloff(1.9e29, -7.8, 0)
    bi = "m3 mol-1 s-1"
    hz = "s-1"
    entries = []

    def add(name, ref, units, h, note, slot=None, stock=1.0):
        if slot is not None:
            # The coupled, conserving production kernel and closed-form
            # coefficients are independent computations of the same local limit.
            np.testing.assert_allclose(
                flux[:, slot] / stock, h, rtol=2e-5, atol=2e-10
            )
        entries.append(
            record(
                name,
                "conditional loss frequency",
                ref,
                units,
                h,
                hz,
                BATH,
                note,
            )
        )

    add(
        "R10 KOH + HCl -> KCl + H2O",
        1.7e8,
        bi,
        gate(15) * 1.7e8 * 5 * one_ppm,
        "Per KOH; explicit hydrolysis-pair gate b15.",
    )
    add(
        "R10b KCl + H2O -> KOH + HCl",
        1.7e8 * kh,
        bi,
        gate(15) * 1.7e8 * kh * 0.05 * c,
        "Per KCl; reference reverse coefficient from hydrolysis Kp.",
    )
    add(
        "R11 KOH + SO2 (+M) -> KHSO3",
        k0,
        bi,
        gate(16) * k11 * 50 * one_ppm,
        "Per KOH; atmospheric Lindemann falloff and b16.",
        6,
    )
    add(
        "R11b KHSO3 -> KOH + SO2",
        k0 * 1e5 / (R * 1000 * k11p),
        hz,
        gate(18) * k11 * 1e5 / (R * 1000 * k11p * stab_b),
        "Per KHSO3; thermochemical dissociation and b18.",
        11,
    )
    add(
        "R12 KOH + SO3 (+M) -> KHSO4",
        k0,
        bi,
        gate(17) * k12 * one_ppm,
        "Per KOH; same source falloff as R11, distinct gates b17 and b34.",
        7,
    )
    add(
        "R12b KHSO4 -> KOH + SO3",
        k0 * 1e5 / (R * 1000 * k12p),
        hz,
        gate(20) * k12 * 1e5 / (R * 1000 * k12p * stab_a),
        "Per KHSO4; reverse Kp and b20.",
        19,
    )
    add(
        "R13b KClSO3 -> KCl + SO3",
        k13_base * 1e5 / (R * 1000 * k13p),
        hz,
        gate(19) * k13 * 1e5 / (R * 1000 * k13p * stab_c),
        "Per KClSO3; reverse Kp and b19.",
        13,
    )
    add(
        "R17b KHSO3 + O2 -> KHSO4 + O",
        2e6 * k17p,
        bi,
        2e6 * k17p * stab_a / stab_b * modifier(30) * c * 0.05,
        "Per KHSO3; reverse of estimated R17 via Kp. Radical alternative b28 "
        "is zero.",
        10,
    )
    add(
        "R18 KHSO4 + KOH -> K2SO4 + H2O",
        1e8,
        bi,
        gate(22) * 1e8 * one_ppm,
        "Per KHSO4; KOH attack, b22.",
        16,
    )
    add(
        "R18b K2SO4 + H2O -> KHSO4 + KOH",
        1e8 / k18p,
        bi,
        gate(26) * 1e8 / k18p * stab_a * c * 0.05,
        "Per gaseous K2SO4; water return, b26.",
        9,
    )
    add(
        "R19 KHSO4 + KCl -> K2SO4 + HCl",
        3e7,
        bi,
        3e7 * 200 * one_ppm * (1 + gate(39) * (mono - 1)) * modifier(32),
        "Per KHSO4; gas KCl availability and fitted terminal capture.",
        15,
    )
    add(
        "R19b K2SO4 + HCl -> KHSO4 + KCl",
        3e7 / k19p,
        bi,
        gate(25) * 3e7 * stab_a / k19p * 5 * one_ppm,
        "Per gaseous K2SO4; product return, b25.",
        8,
    )
    add(
        "R20 KClSO3 + H2O -> KHSO4 + HCl",
        1e8,
        bi,
        1e8 * c * 0.05 * modifier(31),
        "Per KClSO3; rapid hydrolysis requires an occupied pool.",
        12,
    )
    add(
        "R20b KHSO4 + HCl -> KClSO3 + H2O",
        1e8 / k20p,
        bi,
        gate(24) * 1e8 * 5 * one_ppm * stab_c / (k20p * stab_a),
        "Per KHSO4; HCl rebinding, b24.",
        17,
    )
    add(
        "R21 KClSO3 + KOH -> K2SO4 + HCl",
        1e8,
        bi,
        gate(23) * 1e8 * one_ppm,
        "Per KClSO3; KOH attack, b23.",
        14,
    )
    # In the reduced model, R13 entry draws on the prompt SO3-capture channel.
    add(
        "R13 proxy KCl + SO3 (+M) -> KClSO3",
        k13_base,
        bi,
        gate(14) * flux[:, 1],
        "Per SO3; b14 times net prompt SO3 capture. Source column is "
        "elementary Lindemann rate.",
        3,
    )
    add(
        "R16 KHSO4 + H -> KHSO3 + OH",
        8.4e3 * 1000**1.22 * exp(-93303.2 / (R * 1000)),
        bi,
        0 * gate(29),
        "Per KHSO4; H bath absent, b29=0. Source estimate evaluated at 1000 K.",
    )
    add(
        "R17 KHSO4 + O -> KHSO3 + O2",
        2e6,
        bi,
        0 * gate(27),
        "Per KHSO4; no imposed O bath, b27=0. Source estimates this direction.",
        18,
    )
    for j, label, stock in [
        (0, "Prompt SO2 wet sulfation", 50.0),
        (1, "Prompt SO3 capture", 1.0),
    ]:
        entries.append(
            record(
                label,
                "effective substrate loss frequency",
                "",
                "",
                flux[:, j] / stock,
                hz,
                BATH,
                "Per SO2 or SO3 before downstream diversion; includes "
                "equilibrium and legacy gates.",
            )
        )
    offsets = 7.5 * x[:, 13]
    entries[-2]["interpretation"] += (
        f" The inhibition term adds {offsets[0]:.3f}/{offsets[1]:.3f} ppm "
        "to free HCl (56/85 fits); this is an effective offset."
    )
    # The R10 ledger stores net hydrolysis; a longer probe avoids loss of
    # precision on subtracting its two nearly unchanged 1 ppm reservoirs.
    net_hydrolysis = gate(15) * 1.7e8 * c * (kh * 0.05 * 200 - 5e-6)
    net_flux, _ = probe(x, 1e-11)
    np.testing.assert_allclose(
        net_flux[:, 5], net_hydrolysis, rtol=0.003, atol=5e-6
    )
    return annotate_wet(entries, c, hz)


def annotate_wet(entries, c, hz):
    """Convert source coefficients to hazards for the specified bath."""
    # Partner factors convert unmodified source coefficients to comparable
    # first-order rates at the same bath. For the R13 proxy the substrate is
    # SO3. Either supplied analytical KCl or a monomer-only basis changes
    # the comparison, hence no reference hazard is printed for that proxy.
    partners = {
        "R10": 5e-6,
        "R10b": 0.05,
        "R11": 50e-6,
        "R12": 1e-6,
        "R17b": 0.05,
        "R18": 1e-6,
        "R18b": 0.05,
        "R19": 200e-6,
        "R19b": 5e-6,
        "R20": 0.05,
        "R20b": 5e-6,
        "R21": 1e-6,
        "R16": 0.0,
        "R17": 0.0,
    }
    reverse = {"R10b", "R11b", "R12b", "R13b", "R17b", "R18b", "R19b", "R20b"}
    for entry in entries:
        key = entry["reaction"].split()[0]
        if key in reverse:
            origin = (
                "Study detailed balance from source coefficient and study "
                "equilibrium fit"
            )
        elif key in {"R11", "R12", "R13"}:
            origin = (
                "Chanpirak Table 2 high/low limits combined by study "
                "Lindemann formula"
            )
        elif key == "Prompt":
            origin = (
                "No single elementary coefficient for this effective channel"
            )
        else:
            origin = (
                "Chanpirak Table 2 mechanism coefficient (includes estimates "
                "and analogues)"
            )
        entry["reference_origin"] = origin
        if entry["reference_units"] == hz:
            entry["reference_hazard_at_bath_s1"] = entry[
                "reference_coefficient"
            ]
        elif key in partners:
            entry["reference_hazard_at_bath_s1"] = (
                entry["reference_coefficient"] * partners[key] * c
            )
    return entries


def dry_rows(x):
    """Read dry coefficients and check the original rate laws."""
    idx = int(
        np.flatnonzero((dry.T == 1273) & (dry.G["oxygen_percent"] == 2.6))[0]
    )
    rc, offset = dry._inputs(x[:, 29:], 12, 0)
    base, _ = dry._inputs(np.zeros((1, 40)), 12, 0)
    temp = dry.T[idx]
    pressure = dry.PRESSURE_BAR * 1e5
    c = pressure / (R * temp)
    n2 = c * (1 - 0.026 - (500 + 420 + 100) * 1e-6)
    source1 = 5.7e11 * exp(-40000 / temp)
    source2 = 7.8e5 * exp(-3065 / temp)
    np.testing.assert_allclose(
        [base[0, idx, 0], base[0, idx, 2]],
        [source1 * n2, source2 * c * 1e-6],
        rtol=1e-12,
    )
    bath = (
        f"2.6% O2; 500 ppm SO2, 420 ppm SO3; inferred H2O "
        f"{offset[0, idx, 6]:.6g}/{offset[1, idx, 6]:.6g} ppm (split/all); "
        "N2 balance"
    )
    rows = [
        record(
            "Dry R1b SO3 + N2 -> SO2 + O + N2",
            "conditional SO3 loss frequency",
            source1,
            "m3 mol-1 s-1",
            rc[:, idx, 0],
            "s-1",
            bath,
            "Includes collider and fit modifiers. Baseline H2O is 100 ppm.",
            DRY,
            temp,
            pressure,
        ),
        record(
            "Dry R2 SO3 + O -> SO2 + O2",
            "effective bimolecular coefficient",
            source2,
            "m3 mol-1 s-1",
            rc[:, idx, 2] / dry.U[idx],
            "m3 mol-1 s-1",
            bath,
            "Solver uses k times molar density per ppm. Multiply by local "
            "[O] for a hazard; includes competing rate interpolation.",
            DRY,
            temp,
            pressure,
        ),
    ]
    rows[0]["reference_origin"] = (
        "High-temperature SO3 dissociation low-pressure expression"
    )
    rows[0]["reference_hazard_at_bath_s1"] = base[0, idx, 0]
    rows[1]["reference_origin"] = (
        "Original dry abstraction law before interpolation to the "
        "alternative rate"
    )
    return rows


def cluster_rows(u):
    """First-association coefficients before evaporation and survival."""
    collision = 6e-10 * sqrt(1000 / 900)
    rows = []
    for j, core in enumerate(
        ("KCl(g) + K2SO4(g)", "KCl(g) + KHSO4(g)", "ion + K2SO4(g)")
    ):
        values = collision * u[:, 6] * u[:, j]
        rows.append(
            record(
                core,
                "initial binary encounter coefficient",
                collision,
                "cm3 molecule-1 s-1",
                values,
                "cm3 molecule-1 s-1",
                BATH,
                "Per collision partner (per ion for charge). Subsequent "
                "sulfate addition, evaporation and size commitment determine "
                "births.",
                "scripts/straw_combustion/forward/ladders.py",
            )
        )
        rows[-1]["reference_origin"] = (
            "Study assumed collision scale before attachment and pathway "
            "efficiencies"
        )
    return rows


def write_modifiers(fits):
    """Join the tested rate forms to their fitted coordinates in study order."""
    with (HERE / "inputs/parameter_map.csv").open(newline="") as stream:
        mapped = list(csv.DictReader(stream))
    rows = []
    for term in mapped:
        index = int(term["index"])
        if term["vector"] != "x" or index < 9:
            continue
        first_series = index < 29
        rows.append(
            {
                "series": "initial 20" if first_series else "network 40",
                "round": index - (8 if first_series else 28),
                "x_index": index,
                "process": term["label"],
                "tested_form": term["sourced_form"],
                "lower_bound": term["tested_low"],
                "upper_bound": term["tested_high"],
                "split_coordinate": fits["split_combined"]["x"][index],
                "all_data_coordinate": fits["all_combined"]["x"][index],
                "source": term["source_url"],
            }
        )
    with OUT.with_name("reaction-modifiers.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Write the reference-state comparison and verify its short-step limit."""
    with (HERE / "inputs/parameters.json").open() as f:
        fits = json.load(f)
    x = np.array([fits[name]["x"] for name in FITS])
    u = np.array([fits[name]["u"] for name in FITS])
    rows = wet_rows(x) + dry_rows(x) + cluster_rows(u)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    write_modifiers(fits)
    print(f"Wrote {len(rows)} rate rows and the reaction modifier catalog.")


if __name__ == "__main__":
    main()
