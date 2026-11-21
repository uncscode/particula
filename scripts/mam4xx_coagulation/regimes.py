"""Study atmospheric process effects with matched structures and inputs."""

import time

import numpy as np
from scripts.mam4xx_coagulation.campaign import ensemble
from scripts.mam4xx_coagulation.matched import anchors, matched
from scripts.mam4xx_coagulation.native_regimes import comparison
from scripts.mam4xx_coagulation.regime_box import MODES, evolve

PAIRS = {
    "coarse": ["fine", "brown"],
    "charge": ["neutral", "diagnostic"],
    "ion history": ["diagnostic", "dynamic"],
    "dispersion": ["neutral", "dispersion"],
    "dielectric + dispersion": ["diagnostic", "combined"],
    "settling": ["brown", "settling"],
    "shear": ["brown", "shear"],
}


def regime_options(control, regime, options):
    """Apply external field temperature endpoints in the joint input tests."""
    bounds = {
        "marine": (13.9, 14.6),
        "continental": (-4.0, 17.0),
        "polluted": (8.2, 34.0),
    }
    if control in ("cold/lower", "warm/higher"):
        return options | {
            "temp": 273.15 + bounds[regime][control == "warm/higher"]
        }
    return options


def simulate():
    """Run matched rates, paired sensitivities and native particle repeats."""
    benchmark = anchors()
    for item in benchmark:
        item["replay"] = ensemble(
            item["diameter"], 4.0, 3840, 60.0, range(101, 113)
        )
    records = []
    controls = {
        "central": dict(),
        "grid": dict(per_decade=12, cap=5, rtol=6e-7),
        "weak material": dict(hamaker=4.45e-20, dielectric=2.1),
        "strong material": dict(hamaker=1.78e-19, dielectric=12.0),
        "cold/lower": dict(
            factor=0.7, hamaker=4.45e-20, source=2.0, epsilon=0.001
        ),
        "warm/higher": dict(
            factor=1.3, hamaker=1.78e-19, source=10.0, epsilon=0.1
        ),
    }
    for title, options in controls.items():
        for regime in MODES:
            for mechanism in sorted(set(sum(PAIRS.values(), []))):
                row = evolve(
                    regime, mechanism, **regime_options(title, regime, options)
                )
                row["control"] = title
                records.append(row)
            print("Completed", title, regime, flush=True)
    modal_cases = []
    for title, opt in controls.items():
        for regime, modes in MODES.items():
            start = time.monotonic()
            row = matched(
                1000 * modes[1][0],
                32,
                modes,
                regime_options(title, regime, opt).get("temp", 298.15),
                opt.get("factor", 1.0),
            )
            row["elapsed"] = time.monotonic() - start
            row.update(regime=regime, control=title, mechanism="modal")
            modal_cases.append(row)
    summary = []
    for regime in MODES:
        for label, (ref, test) in (
            PAIRS | {"mode width": ["fine", "modal"]}
        ).items():

            def metrics(key, regime=regime, ref=ref, test=test):
                x = {
                    r["mechanism"]: r
                    for r in records + modal_cases
                    if r["regime"] == regime and r["control"] == key
                }
                # Normalize retention to the initial count to allow for the
                # small addition of initial UF particles from the coarse tail.
                a, b = [np.array(x[m]["uf_cm3"]) for m in (ref, test)]
                return 100 * ((b[-1] / b[0]) / (a[-1] / a[0]) - 1)

            values = [metrics(k) for k in controls]
            summary.append(
                dict(
                    regime=regime,
                    process=label,
                    effect=values[0],
                    low=min(values),
                    high=max(values),
                    grid_shift=abs(values[1] - values[0]),
                )
            )
    return dict(
        atmospheric_reference=comparison(),
        endpoints=endpoint_contrasts(records + modal_cases),
        anchors=benchmark,
        modal_cases=modal_cases,
        cases=records,
        contrasts=summary,
    )


def endpoint_contrasts(rows):
    """Absolute and relative paired effects at each reporting time."""
    result = []
    index = {(r["regime"], r["control"], r["mechanism"]): r for r in rows}
    for regime, control in dict.fromkeys(
        (r["regime"], r["control"]) for r in rows
    ):
        for label, (ref, test) in (
            PAIRS | {"mode width": ["fine", "modal"]}
        ).items():
            pair = [index[regime, control, k] for k in (ref, test)]
            for column in ("uf_cm3", "above_cm3", "volume_above_100_um3_cm3"):
                a, b = (np.array(r[column])[1:] for r in pair)
                result.append(
                    dict(
                        regime=regime,
                        control=control,
                        process=label,
                        quantity=column,
                        hours=[1, 6, 24],
                        difference=(b - a).tolist(),
                        percent=(100 * (b / a - 1)).tolist(),
                    )
                )
    return result
