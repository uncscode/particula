"""Find the smallest synthetic angular mismatch needed to change the interpretation."""

import numpy as np
from scipy.optimize import linprog
from optical_test import (
    DIAM,
    LOG,
    D2,
    I,
    ANGLES,
    NOMINAL,
    scattering,
    spectrum,
    cutrow,
    save,
    read,
)


def min_error(A, f, condition, smooth):
    """Minimize max error per weighted ring with arbitrary signal gain."""
    b = A @ f
    weight = np.maximum(b, 0.01 * b.max())
    A = A / weight[:, None]
    b = b / weight
    n = len(f)
    num = len(b)
    M = np.vstack(
        [
            np.c_[A, -np.ones(num)],
            np.c_[-A, -np.ones(num)],
            np.r_[condition, 0.0],
        ]
    )
    rhs = np.r_[b, -b, 0.0]
    lower = [(0, None)] * (n + 1)
    if smooth:
        # Allow up to twice the reference channel curvature and one adjacent
        # channel beyond each occupied endpoint, confined to diameters <=63 µm.
        ix = np.where(f > 1e-10)[0]
        support = (
            (np.arange(n) >= max(ix[0] - 1, 0))
            & (np.arange(n) <= ix[-1] + 1)
            & (DIAM <= 63.0)
        )
        lower = [(0, None) if keep else (0, 0) for keep in support] + [
            (0, None)
        ]
        m = len(D2)
        M = np.pad(M, ((0, 0), (0, m)))
        M = np.vstack(
            [
                M,
                np.c_[D2, np.zeros(m), -np.eye(m)],
                np.c_[-D2, np.zeros(m), -np.eye(m)],
                np.r_[-2 * sum(abs(D2 @ f)) * np.ones(n), 0.0, np.ones(m)],
            ]
        )
        rhs = np.r_[rhs, np.zeros(2 * m + 1)]
        lower += [(0, None)] * m
    c = np.zeros(len(lower))
    c[n] = 1
    res = linprog(c, A_ub=M, b_ub=rhs, bounds=lower, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x[:n]
    mismatch = max(abs(A @ x - b))
    if abs(mismatch - res.fun) > 1e-5 or condition @ x > 1e-7 or x.sum() <= 0:
        raise ValueError(
            ("Endpoint feasibility failed", res.fun, mismatch, condition @ x)
        )
    return max(res.fun, mismatch)


def run():
    out = []
    fe = spectrum(2, "early_pct")
    fl = spectrum(2, "measured_pct")
    f6 = spectrum(6, "measured_pct")
    mass_row = next(r for r in read("total-mass.csv") if r["wind_ms"] == "2")
    m0, m3 = float(mass_row["early_mg"]), float(mass_row["measured_mg"])
    band = cutrow(20) - cutrow(10)
    below63 = cutrow(63)
    ratio6 = next(
        float(r["threshold_mass_share_pct"]) / 100
        for r in read("consistency.csv")
        if r["wind_ms"] == "6"
        and r["time_h"] == "3.0"
        and r["cut_um"] == "20.0"
    )
    for order, diameter_order in ((36, 3), (72, 3), (72, 12), (72, 24)):
        full = scattering(NOMINAL.real, NOMINAL.imag, order, diameter_order)
        for maskname, mask in [
            ("full", np.ones(len(ANGLES) - 1, bool)),
            ("limited", (ANGLES[:-1] >= 1) & (ANGLES[1:] <= 65)),
        ]:
            A = full[mask]
            for prior in (False, True):
                low, high = 6.0, 11.0
                for _ in range(23):
                    mid = (low + high) / 2
                    c = cutrow(mid) - 0.5
                    # early F(mid)<=0.5 and late F(mid)>=0.5 allow loss of the upward shift.
                    a = min_error(A, fe, c, prior)
                    b = min_error(A, fl, -c, prior)
                    if max(a, b) < 1e-7:
                        break
                    if a > b:
                        high = mid
                    else:
                        low = mid
                f_err = min_error(A, f6, cutrow(20) - ratio6 * below63, prior)
                low, high = 0.0, min(m0, m3)
                for _ in range(24):
                    midmass = (low + high) / 2
                    ea = min_error(A, fe, midmass / m0 * below63 - band, prior)
                    la = min_error(A, fl, band - midmass / m3 * below63, prior)
                    if max(ea, la) < 1e-7:
                        break
                    if ea > la:
                        high = midmass
                    else:
                        low = midmass
                out.append(
                    dict(
                        coverage=maskname,
                        quadrature=order,
                        diameter_quadrature=diameter_order,
                        shape_prior=prior,
                        common_band_mass_mg=midmass,
                        min_error_pct_to_erase_band_gain=100 * max(ea, la),
                        threshold_d_um=mid,
                        min_error_pct_for_2ms_sign_loss=100 * max(a, b),
                        min_error_pct_for_6ms_threshold_ratio=100 * f_err,
                    )
                )
    save("optical-tipping.csv", out)


if __name__ == "__main__":
    run()
