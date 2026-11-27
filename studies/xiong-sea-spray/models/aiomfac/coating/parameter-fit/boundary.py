"""Polish incipient two-liquid coexistence by minimizing the drying-path feed molality."""

import numpy as np
from scipy.optimize import minimize, brentq
from trial import states, MW_WATER
from stability import grid


def evaluate(model, salt, m, u, v, temp):
    return states(model, salt, 1 / (MW_WATER * np.asarray(m)), u, v, temp)


def cloud(model, salt, ov, temp=275.15, rho=1.266, cap=160.0, refine=2, details=False):
    g = grid(salt, ov, temp, model, refine=refine, maxmol=cap, density=rho)
    frac = g.fraction
    na = frac if salt == "AS" else 1.0
    best = np.full(len(g.molality), np.inf)
    witness = np.zeros(len(best), dtype=int)
    for i in range(0, len(g.energies), 8192):
        sl = slice(i, i + 8192)
        vals = g.energies[sl, None] - g.trial_coordinates[sl] @ g.feed_mu.T
        indices = np.argmin(vals, axis=0)
        small = vals[indices, np.arange(len(best))]
        take = small < best
        best[take] = small[take]
        witness[take] = i + indices[take]
    hit = np.flatnonzero(best < -1e-6)
    if not len(hit):
        return (np.nan,) * (5 if details else 3)
    if hit[0] == 0:
        return (np.nan,) * (5 if details else 3)

    def tpd(x):
        mf, mt = np.exp(x[:2])
        v = x[-1]
        u = x[2] if salt == "AS" else 1.0
        c, mu = evaluate(model, salt, [mf, mt], [na, u], [frac, v], temp)
        return c[1] @ (mu[1] - mu[0])

    def tpd_grad(x):
        mf, mt = np.exp(x[:2])
        v = x[-1]
        u = x[2] if salt == "AS" else 1.0
        step = 1e-5
        c, mu = evaluate(
            model,
            salt,
            [mf * np.exp(-step), mf * np.exp(step), mf, mt],
            [na, na, na, u],
            [frac, frac, frac, v],
            temp,
        )
        dmu = mu[3] - mu[2]
        grad = [-c[3] @ (mu[1] - mu[0]) / (2 * step), -c[3, 0] * dmu[0]]
        if salt == "AS":
            grad += [dmu[1] - dmu[2], dmu[3] - 0.5 * dmu[4]]
        else:
            grad += [dmu[3] - dmu[2]]
        return np.array(grad)

    j0 = hit[0]
    results = []
    # Multiple starts and distinct modes from the wet side of the phase region.
    for j in np.unique(hit[np.minimum([0, 1, 3, 8], len(hit) - 1)]):
        c = g.trial_coordinates[witness[j]]
        mt = 1 / (MW_WATER * c[0])
        os = c[1 + model.ions.index(301)]
        x0 = [np.log(g.molality[j]), np.log(mt)]
        if salt == "AS":
            x0.append(c[1])
        x0.append(os)
        bounds = [
            (np.log(g.molality[j]) - 0.35, np.log(g.molality[j]) + 0.08),
            (np.log(0.015), np.log(cap)),
        ] + [(1e-9, 1 - 1e-9)] * (len(x0) - 2)
        x0 = np.asarray(x0)
        x0[2:] = np.clip(x0[2:], 1e-9, 1 - 1e-9)

        def constraint(x):
            return -tpd(x) - 1e-7

        if constraint(x0) > 0:
            results.append(x0.copy())
            lo = x0.copy()
            lo[0] = np.log(g.molality[j0 - 1])
            if constraint(lo) < 0:
                anchor = x0.copy()
                anchor[0] = brentq(lambda a: constraint(np.r_[a, x0[1:]]), lo[0], x0[0])
                results.append(anchor)
        opt = minimize(
            lambda x: x[0],
            x0,
            jac=lambda x: np.eye(len(x))[0],
            bounds=bounds,
            constraints=[dict(type="ineq", fun=constraint, jac=lambda x: -tpd_grad(x))],
            method="SLSQP",
            options=dict(ftol=1e-10, maxiter=120),
        )
        if opt.success and tpd(opt.x) < -5e-8:
            results.append(opt.x)
    if results:
        # Alternate a fixed-feed tangent minimization with a wet-side root.
        # Near a narrow two-liquid window SLSQP can collapse onto the trivial phase.
        anchor = min(results, key=lambda r: r[0]).copy()
        for _ in range(18):
            fnow = anchor[0]

            def fixed(x):
                y = np.r_[fnow, x]
                return tpd(y), tpd_grad(y)[1:]

            opt = minimize(
                fixed,
                anchor[1:],
                jac=True,
                method="L-BFGS-B",
                bounds=bounds[1:],
                options=dict(ftol=1e-15, gtol=1e-9, maxiter=90),
            )
            if opt.fun >= -1.00001e-7:
                break
            trial = opt.x

            def residual(z):
                return tpd(np.r_[z, trial]) + 1e-7

            lo = fnow - 0.02
            while lo > np.log(0.01) and residual(lo) < 0:
                lo -= 0.02
            if residual(lo) < 0:
                break
            new = brentq(residual, lo, fnow)
            anchor = np.r_[new, trial]
            results.append(anchor)
            if fnow - new < 1e-8:
                break
    if not results:
        raise RuntimeError(f"No continuous onset for {salt}, {ov} at {g.molality[j0]}")
    x = min(results, key=lambda r: r[0])
    mf, mt = np.exp(x[:2])
    _, mu = evaluate(model, salt, mf, na, frac, temp)
    out = (100 * np.exp(mu[0, 0]), mf, mt)
    if details:
        return (*out, float(frac), float(x[-1]))
    return out


def state_at_rh(model, salt, ov, rh=85.0, temp=275.15, rho=1.266, cap=160.0):
    g = grid(salt, ov, temp, model, refine=2, maxmol=cap, density=rho)
    f = g.fraction
    na = f if salt == "AS" else 1.0

    def feed(m):
        return evaluate(model, salt, m, na, f, temp)

    ms = np.geomspace(0.015, cap, 801)
    tar = np.log(rh / 100)
    y = feed(ms)[1][:, 0] - tar
    roots = [
        brentq(lambda m: feed(m)[1][0, 0] - tar, lo, hi)
        for lo, hi in zip(ms[:-1][y[:-1] * y[1:] < 0], ms[1:][y[:-1] * y[1:] < 0])
    ]
    if not roots:
        return np.nan, np.nan

    def potential(m):
        c, mu = feed(m)
        return c[0] @ mu[0] - c[0, 0] * tar

    m = min(roots, key=potential)
    _, mu = feed(m)
    return m, min(0.0, np.min(g.energies - g.trial_coordinates @ mu[0]))
