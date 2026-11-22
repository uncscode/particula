"""Audit all 15 archived SiMBA profiles against their published generators.

Run with the authors' unmodified GitHub source tarball as the sole argument.
The random sequence, clipping and RK45 defaults reproduce their observation
protocol. Concentrations and time use each source script's numerical scale.
"""

import hashlib
import json
import sys
import tarfile

import numpy as np
from scipy.integrate import solve_ivp

REVISION = "8f0d16357193d79ea73f517540571fc6997f0941"


def hypothetical(t, y, balanced=False):
    """Upstream hypothetical ODE, optionally correcting dimer stoichiometry."""
    a, _, _, d, e = y
    dimer, first, second, last = 0.1 * a * a, 0.2 * a, 0.13 * d, 0.25 * a * e
    return [
        -(1 + balanced) * dimer - first - last,
        dimer,
        last,
        first - second,
        second - last,
    ]


def aldol(t, y):
    """The three extents of the authors' aldol benchmark."""
    a, b, _, _, e, f = y
    first, add, release = 0.7593 * a, 0.2928 * e * b, 0.6813 * f
    return [-first, -add, release, release, first - add, add - release]


def fructose(t, y):
    """Isothermal empirical rate used to generate the paper's HMF profiles."""
    flux = 0.9 * np.exp(-124 / (8.314 * (137 + 273.15))) * 0.033 * y[0]
    return [-flux, 3 * flux, flux]


CASES = {
    "hypoth": (
        hypothetical,
        10,
        0.15,
        3,
        [
            [10, 0, 2, 0, 0],
            [10, 2, 0, 0, 0],
            [10, 2, 2, 0, 0],
            [5, 0, 0, 0, 0],
            [10, 0, 0, 0, 0],
        ],
    ),
    "aldol_condensation": (
        aldol,
        10,
        0.15,
        4,
        [
            [5, 10, 0, 0, 0, 0],
            [5, 5, 2, 0, 0, 0],
            [5, 10, 0, 2, 0, 0],
            [10, 10, 0, 2, 0, 0],
            [10, 10, 2, 2, 0, 0],
        ],
    ),
    "fruc_HMF2": (
        fructose,
        90,
        0.2,
        3,
        [[4, 0, 0], [6, 2, 1], [4, 2, 0], [6, 0, 0], [6, 2, 0]],
    ),
}


def audit(path):
    """Return input digests and maximum deviations for every source run."""
    result = {}
    with tarfile.open(path) as archive:
        for case, (rhs, duration, sigma, observed, initials) in CASES.items():
            rng = np.random.RandomState(1998)
            for i, initial in enumerate(initials, 1):
                filename = f"exp_data_{case}/exp_{i}.csv"
                member = f"auto_react_mech_construct-{REVISION}/{filename}"
                raw = archive.extractfile(member).read()
                actual = np.loadtxt(raw.splitlines(), delimiter=",", skiprows=1)
                times = np.linspace(0, duration, 30)
                solution = solve_ivp(rhs, (0, duration), initial, t_eval=times)
                noise = rng.normal(0, sigma, solution.y.shape)
                expected = np.maximum(solution.y + noise, 0)[:observed].T
                np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-7)
                result[filename] = {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "max_absolute_difference": float(
                        np.max(abs(expected - actual))
                    ),
                }
                if rhs is hypothetical:
                    corrected = solve_ivp(
                        rhs, (0, duration), initial, args=(True,), t_eval=times
                    )
                    result[filename]["max_stoichiometry_effect_noiseless"] = (
                        float(
                            np.max(
                                abs(
                                    corrected.y[:observed]
                                    - solution.y[:observed]
                                )
                            )
                        )
                    )
    return {"source_revision": REVISION, "experiments": result}


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=2, allow_nan=False))
