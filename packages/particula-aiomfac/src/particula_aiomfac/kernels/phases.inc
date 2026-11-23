// Fixed-inventory phase stability. All amounts and derivatives use independent
// analytical formula units. Each inner chemistry call equilibrates reactions.
// Work slices are owned by (binding, stream, row); no persistent warm starts.
const double infinity = __longlong_as_double(0x7ff0000000000000LL);
double *feed = solver.data + row * solver.shape[1];
double *trial = feed + D, *reference_mu = trial + D, *muA = reference_mu + D,
       *muB = muA + D;
double *xA = muB + D, *xB = xA + D, *z = xB + D, *znew = z + D;
double *grad = znew + D, *gnew = grad + D, *direction = gnew + D, *y = direction + D;
double *hy = y + D, *seed = hy + D, *best = seed + D, *fraction = best + D;
double *inverse = fraction + D;
double *result = split.data + row * split.shape[1];
double sum = 0.;
int present = 0;
chemistry(mass);
if (status.data[row] < 0)
    return;
if (carbonate && !gibbs_carbonate &&
    (initial[co2] > 0. || initial[hc] > 0. || initial[co] > 0.)) {
    status.data[row] = -15;
    return;
}
if (out_solid[0] > 0.) {
    status.data[row] = -10;
    return;
}
double scale_mass = 0.;
for (int j = 0; j < D; ++j)
    scale_mass = fmax(scale_mass, mass[j]);
for (int j = 0; j < D; ++j) {
    feed[j] = mass[j] / scale_mass / formula_mw.data[j];
    sum += feed[j];
    present += feed[j] > 0.;
    result[j] = 1.;
}
result[D] = result[D + 1] = 0.;
// The charge-balanced pair (salt) incidence matrix has ion-count minus
// one independent columns precisely when its occupied ion graph is connected.
// Otherwise independent ion exchange would require undeclared analytical pairs.
int firstIon = -1;
for (int k = 0; k < I; ++k) {
    amounts[N + k] = 0.;
    if (initial[N + k] > 0.)
        firstIon = k;
}
if (!ionic_basis && firstIon >= 0) {
    amounts[N + firstIon] = 1.;
    for (int sweep = 0; sweep < I; ++sweep) {
        bool changed = false;
        for (int j = 0; j < D; ++j)
            if (feed[j] > 0.) {
                bool connected = false;
                for (int k = 0; k < I; ++k)
                    if (conversion[j * (N + I) + N + k] > 0. && amounts[N + k] > 0.)
                        connected = true;
                if (connected)
                    for (int k = 0; k < I; ++k)
                        if (conversion[j * (N + I) + N + k] > 0. && amounts[N + k] == 0.) {
                            amounts[N + k] = 1.;
                            changed = true;
                        }
            }
        if (!changed)
            break;
    }
    for (int k = 0; k < I; ++k)
        if (initial[N + k] > 0. && amounts[N + k] == 0.) {
            status.data[row] = -14;
            return;
        }
}
if (present == 1)
    return;
for (int j = 0; j < D; ++j)
    feed[j] /= sum;

// Evaluate the chemical potential of each supplied neutral molecule or
// charge-balanced formula unit. Molality standards cancel between liquids.
auto potentials = [&](const double *xv, double *mu) __attribute__((noinline)) {
    for (int j = 0; j < D; ++j)
        trial[j] = xv[j] * formula_mw.data[j];
    chemistry(trial);
    if (status.data[row] < 0)
        return false;
    double solv = 0.;
    for (int k = 0; k < N; ++k)
        solv += out_moles[k] * mw[k];
    for (int j = 0; j < D; ++j) {
        double v = 0.;
        if (feed[j] > 0.)
            for (int k = 0; k < N + I; ++k) {
                double nu = conversion[j * (N + I) + k] * formula_mw.data[j];
                if (nu == 0.)
                    continue;
                double loga = 0.;
                if (k < N) {
                    loga =
                        out_neutral_log[k] + log(carbonate && !gibbs_carbonate && k == co2
                                                     ? out_moles[k] / solv
                                                     : out_x[k]);
                } else
                    loga = out_ion_log[k - N] + log(out_mol[k - N]);
                v += nu * loga;
            }
        mu[j] = v;
        if (!isfinite(v))
            return false;
    }
    return true;
};
if (!potentials(feed, reference_mu)) {
    status.data[row] = -10;
    return;
}
double homogeneous = 0.;
for (int j = 0; j < D; ++j)
    homogeneous += feed[j] * reference_mu[j];

// The reciprocal-salt basis admits exchange of each actual ion. A scalar
// charge multiplier projects every trial onto the electroneutral manifold.
double *valence = inverse + D * D;
int charged_anchor = -1, solvent_anchor = -1;
for (int j = 0; j < D; ++j) {
    valence[j] = 0.;
    if (ionic_basis && feed[j] > 0.) {
        for (int k = 0; k < I; ++k)
            valence[j] += conversion[j * (N + I) + N + k] * formula_mw.data[j] * charge[k];
        if (fabs(valence[j]) > .5 &&
            (charged_anchor < 0 ||
             fabs(valence[j]) * feed[j] >
                 fabs(valence[charged_anchor]) * feed[charged_anchor]))
            charged_anchor = j;
        if (valence[j] == 0. && (solvent_anchor < 0 || feed[j] > feed[solvent_anchor]))
            solvent_anchor = j;
    }
}
if (ionic_basis) {
    double net = 0.;
    for (int j = 0; j < D; ++j)
        net += feed[j] * valence[j];
    if (fabs(net) > 5e-12) {
        status.data[row] = -16;
        return;
    }
}
auto logistic = [&](double x) {
    double v = exp(-fabs(x));
    return x >= 0. ? 1. / (1. + v) : v / (1. + v);
};
auto multiplier = [&](const double *coord, bool coexist) {
    if (charged_anchor < 0)
        return 0.;
    // The charge residual is monotone increasing with the multiplier.
    // Ends exceed all bounded optimizer logits on both charge sides.
    double lo = -100., hi = 100.;
    for (int iteration = 0; iteration < 56; ++iteration) {
        double shift = .5 * (lo + hi), net = 0.;
        for (int j = 0; j < D; ++j)
            if (valence[j] != 0.) {
                double v = coord[j] + shift * valence[j];
                net += valence[j] * (coexist ? feed[j] * logistic(v) : exp(v));
            }
        if (net > 0.)
            hi = shift;
        else
            lo = shift;
    }
    return .5 * (lo + hi);
};
// TPD uses softmax coordinates. The split objective uses the component
// fractions in alpha, parameterized by log odds to conserve both inventories.
auto objective = [&](bool coexist, const double *coord,
                     double *g) __attribute__((noinline)) {
    double e = 0.;
    double shift = multiplier(coord, coexist);
    if (!coexist) {
        double vmax = -1.e300, den = 0.;
        for (int j = 0; j < D; ++j)
            if (feed[j] > 0.)
                vmax = fmax(vmax, coord[j] + shift * valence[j]);
        for (int j = 0; j < D; ++j) {
            xA[j] = feed[j] > 0. ? exp(coord[j] + shift * valence[j] - vmax) : 0.;
            den += xA[j];
        }
        for (int j = 0; j < D; ++j)
            xA[j] /= den;
        if (!potentials(xA, muA))
            return infinity;
        for (int j = 0; j < D; ++j)
            e += xA[j] * (muA[j] - reference_mu[j]);
        double projection = 0., scale = 0.;
        for (int j = 0; j < D; ++j) {
            projection += xA[j] * valence[j] * (muA[j] - reference_mu[j]);
            scale += xA[j] * valence[j] * valence[j];
        }
        if (scale > 0.)
            projection /= scale;
        int anchor = -1;
        for (int j = 0; j < D; ++j) {
            g[j] = xA[j] * (muA[j] - reference_mu[j] - e - valence[j] * projection);
            if (feed[j] > 0. && valence[j] == 0.)
                anchor = j;
        }
        if (ionic_basis && solvent_anchor >= 0)
            anchor = solvent_anchor;
        if (anchor >= 0)
            g[anchor] = 0.;
        if (charged_anchor >= 0)
            g[charged_anchor] = 0.;
    } else {
        for (int j = 0; j < D; ++j) {
            fraction[j] = logistic(coord[j] + shift * valence[j]);
            xA[j] = feed[j] * fraction[j];
            xB[j] = feed[j] * (1. - fraction[j]);
        }
        if (!potentials(xA, muA) || !potentials(xB, muB))
            return infinity;
        double projection = 0., scale = 0.;
        for (int j = 0; j < D; ++j) {
            double w = xA[j] * (1. - fraction[j]);
            projection += w * valence[j] * (muA[j] - muB[j]);
            scale += w * valence[j] * valence[j];
        }
        if (scale > 0.)
            projection /= scale;
        for (int j = 0; j < D; ++j) {
            e += xA[j] * muA[j] + xB[j] * muB[j];
            g[j] = xA[j] * (1. - fraction[j]) * (muA[j] - muB[j] - valence[j] * projection);
        }
    }
    if (charged_anchor >= 0)
        g[charged_anchor] = 0.;
    return e;
};
// Full inverse BFGS with a descent line search. The D-by-D state is
// allocated once, independent of the number of accepted simulation steps.
auto optimize = [&](bool coexist) __attribute__((noinline)) {
    double e = objective(coexist, z, grad);
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j)
            inverse[i * D + j] = i == j;
    for (int it = 0; it < 300; ++it) {
        if (!isfinite(e))
            return e;
        double slope = 0., norm = 0., maxdir = 0.;
        for (int i = 0; i < D; ++i) {
            direction[i] = 0.;
            for (int j = 0; j < D; ++j)
                direction[i] -= inverse[i * D + j] * grad[j];
            norm = fmax(norm, fabs(grad[i]));
            maxdir = fmax(maxdir, fabs(direction[i]));
            slope += direction[i] * grad[i];
        }
        if (norm < 2.e-10)
            return e;
        if (!(slope < 0.)) {
            slope = 0.;
            maxdir = norm;
            for (int i = 0; i < D; ++i) {
                direction[i] = -grad[i];
                slope -= grad[i] * grad[i];
            }
        }
        double step = fmin(1., 6. / fmax(maxdir, 1.e-30)), enew;
        bool accepted = false;
        for (int line = 0; line < 40; ++line) {
            for (int i = 0; i < D; ++i)
                znew[i] = fmin(36., fmax(-36., z[i] + step * direction[i]));
            enew = objective(coexist, znew, gnew);
            if (isfinite(enew) &&
                enew <= e + 1.e-4 * step * slope + 2.e-15 * (1. + fabs(e))) {
                accepted = true;
                break;
            }
            step *= .5;
        }
        if (!accepted)
            return norm < 1.e-7 ? objective(coexist, z, grad) : infinity;
        double sy = 0., yhy = 0., move = 0.;
        for (int i = 0; i < D; ++i) {
            direction[i] = znew[i] - z[i];
            y[i] = gnew[i] - grad[i];
            sy += direction[i] * y[i];
            move = fmax(move, fabs(direction[i]));
        }
        if (sy > 1.e-20) {
            for (int i = 0; i < D; ++i) {
                hy[i] = 0.;
                for (int j = 0; j < D; ++j)
                    hy[i] += inverse[i * D + j] * y[j];
                yhy += y[i] * hy[i];
            }
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j) {
                    inverse[i * D + j] +=
                        (1. + yhy / sy) * direction[i] * direction[j] / sy -
                        (hy[i] * direction[j] + hy[j] * direction[i]) / sy;
                }
        }
        for (int i = 0; i < D; ++i) {
            z[i] = znew[i];
            grad[i] = gnew[i];
        }
        e = enew;
        if (move < 1.e-11)
            return norm < 1.e-7 ? e : infinity;
    }
    return infinity;
};
auto stability = [&]() __attribute__((noinline)) {
    double min_tpd = 1.e300;
    // Each neutral vertex and each charge-balanced cation/anion vertex
    // defines a distinct enrichment direction for a reciprocal mixture.
    int pairs = charged_anchor >= 0 ? C * (I - C) : 0;
    for (int start = -1; start < D + pairs; ++start) {
        int c = -1, a = -1;
        if (start >= D) {
            c = N + (start - D) / (I - C);
            a = N + C + (start - D) % (I - C);
            if (feed[c] == 0. || feed[a] == 0.)
                continue;
        } else if (start >= 0 && (feed[start] == 0. || valence[start] != 0.))
            continue;
        double tpd = infinity;
        for (int retry = 0; retry < 8 && !isfinite(tpd); ++retry) {
            double enrichment = (start >= D ? .7 : .9) / pow(2., double(retry));
            int anchor = D - 1;
            while (feed[anchor] == 0.)
                --anchor;
            if (ionic_basis && solvent_anchor >= 0)
                anchor = solvent_anchor;
            for (int j = 0; j < D; ++j) {
                double vertex = double(j == start);
                if (start >= D)
                    vertex = (j == c ? -valence[a] : (j == a ? valence[c] : 0.)) /
                             (valence[c] - valence[a]);
                double v =
                    start < 0 ? feed[j] : (1. - enrichment) * feed[j] + enrichment * vertex;
                z[j] = v > 0. ? log(v) : 0.;
            }
            double shift = z[anchor];
            double voltage = charged_anchor < 0
                                 ? 0.
                                 : (z[charged_anchor] - shift) / valence[charged_anchor];
            for (int j = 0; j < D; ++j)
                z[j] -= shift + voltage * valence[j];
            tpd = optimize(false);
        }
        if (!isfinite(tpd))
            return tpd;
        if (tpd < min_tpd) {
            min_tpd = tpd;
            for (int j = 0; j < D; ++j)
                seed[j] = xA[j];
        }
    }
    return min_tpd;
};
double tpd = stability();
if (!isfinite(tpd)) {
    status.data[row] = -11;
    return;
}
result[D] = tpd;
if (tpd >= -1.e-8) {
    chemistry(mass);
    return;
}
// Start from the most unstable trial. Additional complementary allocations
// recover extreme split ratios for trace or strongly excluded ions.
auto resolve_split = [&](int attempt) __attribute__((noinline)) {
    double step = 1.e300;
    for (int j = 0; j < D; ++j)
        if (feed[j] > 0.)
            step = fmin(step, feed[j] / seed[j]);
    if (attempt >= 0) {
        for (int reductions = 0; reductions < 20; ++reductions) {
            for (int j = 0; j < D; ++j) {
                double f = feed[j] > 0. ? .7 * step * seed[j] / feed[j] : .5;
                if (attempt > 0) {
                    // Enrich one neutral while maintaining local charge closure.
                    f = (j == attempt - 1 ? .03 : .85);
                }
                z[j] = log(f / (1. - f));
            }
            if (charged_anchor >= 0) {
                double voltage = z[charged_anchor] / valence[charged_anchor];
                for (int j = 0; j < D; ++j)
                    z[j] -= voltage * valence[j];
            }
            // Negative TPD guarantees a downhill *infinitesimal* phase. Near the
            // binodal its finite lever-rule initialization must be backtracked.
            if (attempt > 0 || objective(true, z, grad) < homogeneous)
                break;
            step *= .2;
        }
        for (int j = 0; j < D; ++j)
            best[j] = z[j];
        double energy = optimize(true);
        if (!isfinite(energy) || energy >= homogeneous) {
            // The extensive gain at phase birth can be below the optimizer's
            // precision. Refine the chemical potentials directly from the
            // downhill tangent-plane seed before declaring coexistence unresolved.
            for (int j = 0; j < D; ++j)
                z[j] = best[j];
        }
    }
    int pivot_component = 0;
    if (attempt < 0) {
        // A vanishing phase is ill-conditioned in independent component
        // log-odds. The minority mole fraction remains well conditioned as
        // it tends to zero. Represent its normalized composition by logits.
        for (int j = 1; j < D; ++j)
            if (seed[j] > seed[pivot_component])
                pivot_component = j;
        for (int j = 0; j < D; ++j)
            z[j] = feed[j] > 0. ? log(seed[j] / seed[pivot_component]) : 0.;
        z[pivot_component] = fmin(.001, .2 * step);
    }

    // Polish equality of component potentials. Minimizing Gibbs alone underweights
    // the residual of a trace constituent or nearly vanished phase.
    auto residual = [&](const double *coord, double *r) __attribute__((noinline)) {
        if (attempt < 0) {
            const double share = coord[pivot_component];
            if (!(share > 0. && share < 1.))
                return infinity;
            double den = 0.;
            for (int j = 0; j < D; ++j) {
                xA[j] = feed[j] == 0. ? 0. : exp(j == pivot_component ? 0. : coord[j]);
                den += xA[j];
            }
            for (int j = 0; j < D; ++j) {
                xA[j] *= share / den;
                xB[j] = feed[j] - xA[j];
                if (feed[j] > 0. && xB[j] <= 0.)
                    return infinity;
                fraction[j] = feed[j] > 0. ? xA[j] / feed[j] : 0.;
            }
            if (!potentials(xA, muA) || !potentials(xB, muB))
                return infinity;
        } else if (!isfinite(objective(true, coord, gnew)))
            return infinity;
        double err = 0.;
        double potential =
            charged_anchor < 0
                ? 0.
                : (muA[charged_anchor] - muB[charged_anchor]) / valence[charged_anchor];
        for (int j = 0; j < D; ++j) {
            r[j] = muA[j] - muB[j] - potential * valence[j];
            err = fmax(err, fabs(r[j]));
        }
        return err;
    };
    double err = residual(z, grad);
    for (int it = 0; it < 24 && err > 1.e-9; ++it) {
        for (int j = 0; j < D; ++j) {
            for (int i = 0; i < D; ++i)
                znew[i] = z[i];
            znew[j] += 1.e-5;
            if (!isfinite(residual(znew, y))) {
                status.data[row] = -12;
                return;
            }
            for (int i = 0; i < D; ++i)
                inverse[i * D + j] = feed[i] > 0. && i != charged_anchor
                                         ? (y[i] - grad[i]) / 1.e-5
                                         : double(i == j);
        }
        for (int i = 0; i < D; ++i)
            direction[i] = -grad[i];
        // Pivoted Gaussian elimination on the log-odds Jacobian.
        for (int j = 0; j < D; ++j) {
            int pivot = j;
            for (int i = j + 1; i < D; ++i)
                if (fabs(inverse[i * D + j]) > fabs(inverse[pivot * D + j]))
                    pivot = i;
            if (fabs(inverse[pivot * D + j]) < 1.e-14) {
                status.data[row] = -12;
                return;
            }
            for (int k = j; k < D; ++k) {
                double tmp = inverse[j * D + k];
                inverse[j * D + k] = inverse[pivot * D + k];
                inverse[pivot * D + k] = tmp;
            }
            double tmp = direction[j];
            direction[j] = direction[pivot];
            direction[pivot] = tmp;
            for (int i = j + 1; i < D; ++i) {
                double fac = inverse[i * D + j] / inverse[j * D + j];
                for (int k = j + 1; k < D; ++k)
                    inverse[i * D + k] -= fac * inverse[j * D + k];
                direction[i] -= fac * direction[j];
            }
        }
        for (int i = D - 1; i >= 0; --i) {
            for (int k = i + 1; k < D; ++k)
                direction[i] -= inverse[i * D + k] * direction[k];
            direction[i] /= inverse[i * D + i];
        }
        bool accepted = false;
        for (double damping = 1.; damping > 1.e-7; damping *= .5) {
            for (int i = 0; i < D; ++i)
                znew[i] = fmin(36., fmax(-36., z[i] + damping * direction[i]));
            double next = residual(znew, y);
            if (next < err) {
                for (int i = 0; i < D; ++i) {
                    z[i] = znew[i];
                    grad[i] = y[i];
                }
                err = next;
                accepted = true;
                break;
            }
        }
        if (!accepted)
            break;
    }
    err = residual(z, grad);
    double coexist_g = 0.;
    for (int j = 0; j < D; ++j)
        coexist_g += xA[j] * muA[j] + xB[j] * muB[j];
    const double energy_roundoff = 8. * 2.220446049250313e-16 * (1. + fabs(homogeneous));
    double amountA = 0., amountB = 0., composition_gap = 0.;
    for (int j = 0; j < D; ++j) {
        amountA += xA[j];
        amountB += xB[j];
    }
    for (int j = 0; j < D; ++j)
        composition_gap = fmax(composition_gap, fabs(xA[j] / amountA - xB[j] / amountB));
    if (!(err < 1.e-7) || !(composition_gap > 1.e-5) || !isfinite(coexist_g) ||
        coexist_g > homogeneous + energy_roundoff) {
        status.data[row] = -12;
        return;
    }
    for (int j = 0; j < D; ++j) {
        best[j] = fraction[j];
        reference_mu[j] = muA[j];
    }
    tpd = stability();
    if (!isfinite(tpd) || tpd < -1.e-7) {
        status.data[row] = -13;
        return;
    }
    for (int j = 0; j < D; ++j) {
        result[j] = best[j];
        trial[j] = mass[j] * best[j];
    }
    if (gibbs_carbonate && carbonate && ionic_basis) {
        // Check *actual* equilibrated species, including those represented by
        // another analytical carrier. A common electric potential is arbitrary.
        // The independent-ion basis has D=N+I, so grad holds all species here.
        for (int j = 0; j < D; ++j)
            trial[j] = mass[j] * (1. - best[j]);
        chemistry(trial);
        if (status.data[row] < 0)
            return;
        for (int i = 0; i < N; ++i)
            grad[i] = out_neutral_mask[i] ? log(out_x[i]) + out_neutral_log[i] : NAN;
        for (int i = 0; i < I; ++i)
            grad[N + i] = out_ion_mask[i] ? log(out_mol[i]) + out_ion_log[i] : NAN;
        for (int j = 0; j < D; ++j)
            trial[j] = mass[j] * best[j];
        chemistry(trial);
        if (status.data[row] < 0)
            return;
        for (int i = 0; i < N; ++i)
            if (out_neutral_mask[i] && isfinite(grad[i]))
                err = fmax(err, fabs(log(out_x[i]) + out_neutral_log[i] - grad[i]));
        double signed_sum = 0., charge2 = 0.;
        for (int i = 0; i < I; ++i) {
            if (out_ion_mask[i] && isfinite(grad[N + i])) {
                grad[N + i] = log(out_mol[i]) + out_ion_log[i] - grad[N + i];
                signed_sum += grad[N + i] * charge[i];
                charge2 += charge[i] * charge[i];
            } else
                grad[N + i] = NAN;
        }
        for (int i = 0; i < I; ++i)
            if (isfinite(grad[N + i]))
                err = fmax(err, fabs(grad[N + i] - charge[i] * signed_sum / charge2));
        if (!(err <= 1.e-7)) {
            status.data[row] = -12;
            return;
        }
    } else
        chemistry(trial);
    if (status.data[row] >= 0)
        status.data[row] = 2;
    result[D] = tpd;
    result[D + 1] = err;
};
resolve_split(0);
if (status.data[row] < 0 && charged_anchor < 0 && potentials(feed, reference_mu)) {
    const double downhill = stability();
    if (isfinite(downhill) && downhill < -1.e-8)
        resolve_split(-1);
    else
        status.data[row] = -11;
}
for (int j = 0; status.data[row] < 0 && j < D; ++j) {
    if (feed[j] == 0.)
        continue;
    bool neutral = false;
    for (int k = 0; k < N; ++k)
        if (conversion[j * (N + I) + k] > 0.)
            neutral = true;
    if (!neutral)
        continue;
    if (!potentials(feed, reference_mu)) {
        status.data[row] = -10;
        return;
    }
    resolve_split(j + 1);
}
