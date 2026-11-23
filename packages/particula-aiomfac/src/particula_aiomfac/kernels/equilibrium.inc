// Closed analytical inventories are in mol per kg of input. Solve each row on
// this thread so that neither trial compositions nor solver state cross rows.
double solid = 0.;
if (sulfate || carbonate) {
    double sulfur = sulfate ? initial[hs] + initial[so] : 0.;
    const double initial_sulfur = sulfur;
    double initial_cations = 0.;
    for (int i = 0; i < C; ++i)
        initial_cations += initial[N + i];
    if (!gibbs_carbonate && carbonate && sulfur > 0. && ca >= 0) {
        const double calcium = initial[ca];
        double remaining = calcium < sulfur
                               ? sulfur - calcium
                               : fmin(1.e-5 * sulfur, 2.220446049250313e-13 * solvent);
        solid = sulfur - remaining;
        initial[ca] = amounts[ca] = calcium < sulfur ? 0. : calcium - sulfur + remaining;
        initial[h] += initial[hs];
        amounts[h] = initial[h];
        initial[hs] = amounts[hs] = 0.;
        initial[so] = amounts[so] = remaining;
        sulfur = remaining;
    }
    const double carbon = carbonate ? initial[hc] + initial[co] + initial[co2] : 0.;
    const double ts = fmin(473., fmax(180., T)), t0 = 298.15;
    const double cp = -205.594443486, slope = -.994992864240;
    const double dh = -18554.748598 - cp * t0 + .5 * slope * t0 * t0;
    const double ks =
        -4.54916799587 - (dh * (1. / ts - 1. / t0) - (cp - slope * t0) * log(ts / t0) -
                          .5 * slope * (ts - t0)) /
                             8.314472;
    if ((carbonate && gibbs_carbonate) || carbon > 2.220446049250313e-16 * solvent) {
        const bool water_only = carbon == 0.;
        const int n = (water_only ? 2 : 5) + (sulfur > 0. ? 1 : 0);
        const double water_available = gibbs_carbonate ? initial[water] : fmin(3.5 * initial_cations, .98 * initial[water]);
        const double proton_max = initial[h] + initial[hc] + 2. * initial[co2] +
                                  water_available + (sulfate ? initial[hs] : 0.);
        const double bound[5] = {fmin(proton_max, carbon), carbon,
                                 fmin(.5 * proton_max, carbon),
                                 water_available + initial[oh], proton_max};
        const int index[5] = {hc, co, co2, oh, h};
        const double k1 = -820.4327 - .14027266 * T + 50275.49 / T + 126.8339 * log(T) -
                          3879660. / (T * T);
        const double k2 = -248.4192 - .07489962 * T + 11862.43 / T + 38.92561 * log(T) -
                          1297999. / (T * T);
        const double kw =
            -148.16780 + .8933802 * T - .002332199 * T * T + 2.146860e-6 * T * T * T;
        double spectator = 0.;
        for (int j = N; j < S; ++j)
            if (j != h && j != oh && j != hc && j != co &&
                (sulfur == 0. || (j != so && j != hs)))
                spectator += initial[j] * charge[j - N];
        const double norm =
            fmax(1.e-300, fmax(fabs(spectator), fmax(fmax(carbon, sulfur),
                                                     fmax(initial[oh], initial[h]))));
        const double initial_solvent = solvent;
        double seed[6], lo = -50., hi = 100.;
        for (int iter = 0; iter < 64; ++iter) {
            const double ph = .5 * (lo + hi), proton = exp(-ph);
            const double r1 = exp(k1) / proton, r2 = exp(k2) / proton;
            seed[2] = carbon / (1. + r1 * (1. + r2));
            seed[0] = seed[2] * r1;
            seed[1] = seed[0] * r2;
            seed[3] = exp(kw) * initial_solvent / proton;
            seed[4] = proton * initial_solvent;
            seed[5] = proton / exp(ks);
            double net = spectator + seed[4] - seed[3] - seed[0] - 2. * seed[1] -
                         sulfur * (2. + seed[5]) / (1. + seed[5]);
            if (net > 0.)
                lo = ph;
            else
                hi = ph;
        }
        if (water_only) {
            seed[0] = seed[4]; // proton, hydroxide, and optional bisulfate ratio
            seed[1] = seed[3];
            seed[2] = seed[5];
        }
        for (int j = 0; j < n; ++j)
            if (!(seed[j] > 0.) || !isfinite(seed[j])) {
                fail(5);
                return;
            }
        auto equations = [&](const double *z, double *f) __attribute__((noinline)) {
            ++evaluations;
            double v[6];
            for (int j = 0; j < n; ++j)
                v[j] = exp(fmin(50., fmax(-700., z[j])));
            if (water_only) {
                amounts[h] = v[0];
                amounts[oh] = v[1];
            } else {
                amounts[hc] = v[0];
                amounts[co] = v[1];
                amounts[co2] = v[2];
                amounts[oh] = v[3];
                amounts[h] = v[4];
            }
            amounts[water] = initial[water] + (amounts[co2] - initial[co2])
                                            - (amounts[oh] - initial[oh]);
            if (sulfur > 0.) {
                const double ratio = v[n - 1];
                amounts[hs] = sulfur * ratio / (1. + ratio);
                amounts[so] = sulfur / (1. + ratio);
            }
            if (!(amounts[water] > 0.)) {
                for (int j = 0; j < n; ++j)
                    f[j] = 1.e3 + fabs(amounts[water]);
                return;
            }
            terms();
            const double lnaw = lng[water] + log(x[water]), lns = log(solvent);
            const double lh = log(amounts[h]) - lns + lng[h];
            int k = 0;
            if (!water_only) {
                f[k++] = lh + log(v[0]) + lng[hc] - log(v[2]) - co2_molal_log - lnaw - k1;
                f[k++] = lh + log(v[1]) + lng[co] - log(v[0]) - lng[hc] - k2;
            }
            f[k++] = lh + log(amounts[oh]) - lns + lng[oh] - lnaw - kw;
            if (sulfur > 0.)
                f[k++] = lh + lng[so] - lng[hs] - log(v[n - 1]) - ks;
            if (!water_only)
                f[k++] = log((v[0] + v[1] + v[2]) / carbon);
            double net = 0., weight = 0.;
            for (int j = N; j < S; ++j) {
                const double equivalents = amounts[j] * charge[j - N];
                net += equivalents;
                weight += fabs(equivalents);
            }
            f[k] = net / (gibbs_carbonate ? fmax(weight, 1.e-300) : norm);
        };
        // Powell's scaled dogleg in a unit metric. The small dense Jacobian
        // is refreshed per accepted step and factored with partial pivoting.
        bool ok = false;
        const double shifts[5] = {0., .25, -.25, 1., -1.};
        for (int attempt = 0; attempt < 5 && !ok; ++attempt) {
            retries = attempt;
            double z[6], f[6], delta = 0.;
            for (int j = 0; j < n; ++j) {
                z[j] = log(seed[j]) +
                       shifts[attempt] * (j == 5 ? 0. : (j == 2 || j == 3 ? -1. : 1.));
                delta += z[j] * z[j];
            }
            delta = 100. * fmax(1., sqrt(delta));
            int begin = evaluations;
            equations(z, f);
            while (evaluations - begin + n + 1 <= evaluations_per_dimension * (n + 1)) {
                double maximum = 0., cost = 0.;
                for (int i = 0; i < n; ++i) {
                    maximum = fmax(maximum, fabs(f[i]));
                    cost += f[i] * f[i];
                }
                if (isfinite(cost) && maximum < 1.e-12) {
                    ok = true;
                    break;
                }
                double jac[6][6], lu[6][6], newton[6], grad[6] = {}, trial[6], ft[6];
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i)
                        trial[i] = z[i];
                    double step = 1.4901161193847656e-8 * fmax(fabs(z[j]), 1.);
                    trial[j] += step;
                    equations(trial, ft);
                    for (int i = 0; i < n; ++i) {
                        jac[i][j] = lu[i][j] = (ft[i] - f[i]) / step;
                        grad[j] += jac[i][j] * f[i];
                    }
                }
                for (int i = 0; i < n; ++i)
                    newton[i] = -f[i];
                bool singular = false;
                for (int j = 0; j < n; ++j) {
                    int pivot = j;
                    for (int i = j + 1; i < n; ++i)
                        if (fabs(lu[i][j]) > fabs(lu[pivot][j]))
                            pivot = i;
                    if (!(fabs(lu[pivot][j]) > 1.e-25)) {
                        singular = true;
                        break;
                    }
                    for (int k = j; k < n; ++k) {
                        double tmp = lu[j][k];
                        lu[j][k] = lu[pivot][k];
                        lu[pivot][k] = tmp;
                    }
                    double tmp = newton[j];
                    newton[j] = newton[pivot];
                    newton[pivot] = tmp;
                    for (int i = j + 1; i < n; ++i) {
                        double a = lu[i][j] / lu[j][j];
                        for (int k = j + 1; k < n; ++k)
                            lu[i][k] -= a * lu[j][k];
                        newton[i] -= a * newton[j];
                    }
                }
                double ng = 0., njg = 0., nn = 0.;
                for (int i = n - 1; i >= 0; --i) {
                    if (!singular) {
                        for (int j = i + 1; j < n; ++j)
                            newton[i] -= lu[i][j] * newton[j];
                        newton[i] /= lu[i][i];
                        nn += newton[i] * newton[i];
                    }
                    ng += grad[i] * grad[i];
                    double sum = 0.;
                    for (int j = 0; j < n; ++j)
                        sum += jac[i][j] * grad[j];
                    njg += sum * sum;
                }
                double alpha = ng / fmax(1.e-300, njg);
                double step[6];
                if (!singular && sqrt(nn) <= delta)
                    for (int i = 0; i < n; ++i)
                        step[i] = newton[i];
                else if (singular || alpha * sqrt(ng) >= delta)
                    for (int i = 0; i < n; ++i)
                        step[i] = -delta * grad[i] / fmax(1.e-300, sqrt(ng));
                else {
                    double a = 0., b = 0., c = -delta * delta;
                    for (int i = 0; i < n; ++i) {
                        step[i] = -alpha * grad[i];
                        double d = newton[i] - step[i];
                        a += d * d;
                        b += step[i] * d;
                        c += step[i] * step[i];
                    }
                    double beta = (-b + sqrt(fmax(0., b * b - a * c))) / a;
                    for (int i = 0; i < n; ++i)
                        step[i] += beta * (newton[i] - step[i]);
                }
                double predicted = 0., length = 0.;
                for (int i = 0; i < n; ++i) {
                    double v = f[i];
                    for (int j = 0; j < n; ++j)
                        v += jac[i][j] * step[j];
                    predicted += v * v;
                    length += step[i] * step[i];
                    trial[i] = z[i] + step[i];
                }
                equations(trial, ft);
                double next = 0.;
                for (int i = 0; i < n; ++i)
                    next += ft[i] * ft[i];
                double rho =
                    (cost > predicted) ? (cost - next) / (cost - predicted) : -1.;
                if (rho < .25)
                    delta *= .25;
                if (rho > .75)
                    delta = fmax(delta, 2. * sqrt(length));
                if (rho > 1.e-4)
                    for (int i = 0; i < n; ++i) {
                        z[i] = trial[i];
                        f[i] = ft[i];
                    }
                if (delta < 1.4901161193847656e-10)
                    break;
            }
            equations(z, f);
            errors[0] = 0.;
            for (int i = 0; i < n - (water_only ? 1 : 2); ++i)
                errors[0] = fmax(errors[0], fabs(f[i]));
            bool finite_residual = true;
            for (int i = 0; i < n; ++i)
                finite_residual = finite_residual && isfinite(f[i]);
            ok = finite_residual && errors[0] <= 1.e-8 && fabs(f[n - 1]) <= 1.e-10 &&
                 (water_only || fabs(f[n - 2]) <= 1.e-10);
            if (!gibbs_carbonate)
                for (int j = 0; j < 5; ++j)
                    ok = ok && amounts[index[j]] <= bound[j] * (1. + 1.e-10);
            if (!gibbs_carbonate && n == 6)
                ok = ok && amounts[hs] <= fmin(proton_max, sulfur) * (1. + 1.e-10);
        }
        if (!ok) {
            fail(5);
            return;
        }
        if (!water_only)
            errors[2] = fabs(amounts[hc] + amounts[co] + amounts[co2] - carbon) / carbon;
    } else if (fmin(sulfur, initial[h] + initial[hs]) > 2.220446049250313e-19 * solvent) {
        const double hydrogen = initial[h] + initial[hs], bound = fmin(hydrogen, sulfur);
        auto sulfate_equation = [&](double b) __attribute__((noinline)) {
            ++evaluations;
            amounts[hs] = b;
            amounts[so] = sulfur - b;
            amounts[h] = hydrogen - b;
            terms();
            return log(fmax(amounts[h], 1.e-300)) + log(fmax(amounts[so], 1.e-300)) -
                   log(fmax(b, 1.e-300)) - log(solvent) + lng[h] + lng[so] - lng[hs] - ks;
        };
        double lo = 0., hi = bound, b = .5 * bound, y = 0.;
        for (int iter = 0; iter < 100; ++iter) {
            b = .5 * (lo + hi);
            y = sulfate_equation(b);
            if (fabs(y) < 1.e-13 || b == lo || b == hi)
                break;
            if (y > 0.)
                lo = b;
            else
                hi = b;
        }
        if (!(fabs(y) <= 1.e-8)) {
            retries = 1;
            double sum = 0.;
            for (int j = N; j < S; ++j)
                sum += initial[j];
            const double wf = .097071 / (.097071 + 2. * initial[water] * mw[water] / sum);
            const double alpha = fmin(
                1.,
                fmax(0., 1. - pow(1. + pow(wf, -.263468) - pow(wf, -.263204), -73.0039) +
                             1.72067 * pow(wf, .745718) * pow(1. - wf, 1.75)));
            const double scale = 2.220446049250313e-19 * solvent;
            sulfate_equation(alpha > .4 ? alpha * scale : (1. - alpha) * (bound - scale));
            const double v =
                hydrogen + sulfur +
                solvent * exp(ks - fmin(300., fmax(-300., lng[h] + lng[so] - lng[hs])));
            b = 2. * hydrogen * sulfur / (v + sqrt(v * v - 4. * hydrogen * sulfur));
            y = sulfate_equation(b);
        }
        errors[0] = fabs(y);
    }
    double net = 0., weights = 0.;
    for (int j = N; j < S; ++j) {
        double q = amounts[j] * charge[j - N];
        net += q;
        weights += fabs(q);
    }
    errors[1] = fabs(net) / fmax(weights, 1.e-300);
    if (initial_sulfur > 0.)
        errors[3] =
            fabs(amounts[hs] + amounts[so] + solid - initial_sulfur) / initial_sulfur;
    for (int i = 0; i < 4; ++i)
        if (!isfinite(errors[i]) || errors[i] > (i ? 1.e-10 : 1.e-8)) {
            fail(5);
            return;
        }
}
