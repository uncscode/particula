// AIOMFAC-web 3.15 direct activities. One Warp thread owns one mixture row.
// The integer dimensions, parameter views and scratch views precede this snippet.
const int S = N + I, A = I - C;
double maxmass = 0., total = 0.;
auto fail = [&](int code) { status.data[row] = -code; };
if (!(T > 0.) || !isfinite(T)) {
    fail(2);
    return;
}
for (int j = 0; j < D; ++j) {
    if (mass[j] < 0. || !isfinite(mass[j])) {
        fail(1);
        return;
    }
    maxmass = fmax(maxmass, mass[j]);
}
if (maxmass == 0.) {
    fail(1);
    return;
}
for (int j = 0; j < D; ++j)
    total += mass[j] / maxmass;
for (int k = 0; k < S; ++k) {
    initial[k] = 0.;
    for (int j = 0; j < D; ++j)
        initial[k] += mass[j] / maxmass / total * conversion[j * S + k];
    amounts[k] = initial[k];
}
double solvent = 0., strength = 0., scale = 0., mole_sum = 0.;
for (int j = 0; j < N; ++j)
    solvent += amounts[j] * mw[j];
if (!(solvent > 0.) || ((sulfate || carbonate) && !(amounts[water] > 0.))) {
    fail(3);
    return;
}
for (int j = 0; j < G * G; ++j)
    psi[j] = exp(-interaction[j] / T);
// Modified UNIFAC surface fractions, residuals and pure reference values.
auto residual = [&](int pure) __attribute__((noinline)) {
    double sum = 0.;
    for (int g = 0; g < G; ++g) {
        double v = 0.;
        if (pure >= 0)
            v = counts[pure * G + g];
        else
            for (int j = 0; j < S; ++j)
                v += x[j] * counts[j * G + g];
        theta[g] = q[g] * v;
        sum += theta[g];
    }
    for (int g = 0; g < G; ++g)
        theta[g] /= sum;
    for (int j = 0; j < G; ++j) {
        den[j] = 0.;
        for (int g = 0; g < G; ++g)
            den[j] += theta[g] * psi[g * G + j];
    }
    for (int g = 0; g < G; ++g) {
        double v = 0.;
        for (int j = 0; j < G; ++j)
            v += theta[j] / den[j] * psi[g * G + j];
        group[g] = q[g] * (1. - log(den[g]) - v);
    }
};
for (int j = 0; j < N; ++j) {
    residual(j);
    ref[j] = 0.;
    for (int g = 0; g < G; ++g)
        ref[j] += counts[j * G + g] * group[g];
}
auto terms = [&]() __attribute__((noinline)) {
    solvent = 0.;
    mole_sum = 0.;
    double solvent_moles = 0., ion_sum = 0., area_sum = 0., volume_sum = 0.;
    for (int j = 0; j < S; ++j) {
        mole_sum += amounts[j];
        if (j < N) {
            solvent += amounts[j] * mw[j];
            solvent_moles += amounts[j];
        }
    }
    strength = 0.;
    double abs_charge = 0.;
    for (int j = 0; j < I; ++j) {
        mol[j] = amounts[N + j] / solvent;
        strength += 0.5 * mol[j] * charge[j] * charge[j];
        ion_sum += mol[j];
        abs_charge += mol[j] * fabs(charge[j]);
    }
    for (int j = 0; j < S; ++j) {
        x[j] = amounts[j] / mole_sum;
        volume_sum += vol[j] * x[j];
        area_sum += area[j] * x[j];
    }
    residual(-1);
    double root = sqrt(strength), diel = sqrt(78.54 * T);
    double a = 1.327757e5 * sqrt(997.) / (diel * diel * diel),
           b = 6.359696 * sqrt(997.) / diel;
    double f = 1. + b * root;
    for (int j = 0; j < S; ++j) {
        double v = vol[j] / volume_sum;
        double ratio = v * area_sum / area[j];
        sr[j] = log(v) + 1. - v - 5. * area[j] * (log(ratio) + 1. - ratio);
        for (int g = 0; g < G; ++g)
            sr[j] += counts[j * G + g] * group[g];
        if (j < N) {
            sr[j] -= ref[j];
            lr[j] = 2. * a / (b * b * b) * (f - 1. / f - 2. * log(f)) * mw[j];
        } else {
            v = vol[j];
            double qj = area[j];
            sr[j] -= log(v / .92) + 5. * qj * log(qj * .92 / (1.4 * v)) + 5. * (v - qj) -
                     v + 1. + v / .92 * 2.32;
            lr[j] = -a * root / f * charge[j - N] * charge[j - N];
        }
        mr[j] = 0.;
    }
    scale = 0.;
    if (strength > 2.2204460492503131e-16) {
        scale = log(.01801528 * (solvent_moles / solvent + ion_sum));
        double groupsum = 0., avg = 0.;
        for (int g = 0; g < M; ++g) {
            mainx[g] = 0.;
            mainmw[g] = 0.;
        }
        for (int k = 0; k < G - I; ++k) {
            double quantity = 0.;
            for (int j = 0; j < N; ++j)
                quantity += amounts[j] * counts[j * G + k];
            int g = static_cast<int>(main_index[k]);
            mainx[g] += quantity;
            mainmw[g] += quantity * group_mw[k];
            groupsum += quantity;
        }
        double deriv = 0., groupmean = 0.;
        for (int g = 0; g < M; ++g) {
            avg += mainmw[g] / groupsum;
            if (mainx[g] > 0.)
                mainmw[g] /= mainx[g];
            mainx[g] /= groupsum;
            groupion[g] = 0.;
            for (int j = 0; j < I; ++j) {
                double change = cgroup[g * I + j] * exp(-1.2 * root);
                double value = bgroup[g * I + j] + change;
                groupion[g] += value * mol[j];
                mr[N + j] += mainx[g] * value;
                deriv -= .6 / root * change * mol[j] * mainx[g];
            }
            groupmean += mainx[g] * groupion[g];
        }
        for (int j = 0; j < I; ++j)
            mr[N + j] /= avg;
        double pairb = 0., pairc = 0., pd = 0., rsum = 0., qsum = 0.;
        for (int c = 0; c < C; ++c) {
            for (int a = 0; a < A; ++a) {
                int k = c * A + a;
                double xb = c_pair_b[k] * exp(-decay_b[k] * root);
                double xc = c_pair_c[k] * exp(-decay_c[k] * root);
                double B = b_pair_b[k] + xb, K = b_pair_c[k] + xc,
                       prod = mol[c] * mol[C + a];
                pairb += B * prod;
                pairc += K * prod;
                pd -= .5 / root * (decay_b[k] * xb + abs_charge * decay_c[k] * xc) * prod;
                mr[N + c] += (B + abs_charge * K) * mol[C + a];
                mr[N + C + a] += (B + abs_charge * K) * mol[c];
            }
            for (int j = 0; j < C; ++j) {
                double r = like[c * C + j], pair = mol[c] * mol[j];
                mr[N + c] += r * mol[j];
                if (c <= j)
                    rsum += r * pair;
                for (int a = 0; a < A; ++a) {
                    double Q = three[(c * C + j) * A + a];
                    mr[N + c] += Q * mol[j] * mol[C + a];
                    if (c <= j) {
                        mr[N + C + a] += Q * pair;
                        qsum += Q * pair * mol[C + a];
                    }
                }
            }
        }
        for (int g = 0; g < M; ++g) {
            double gr =
                groupion[g] -
                mainmw[g] * ((groupmean + strength * deriv) / avg + pairb +
                             strength * pd + 2. * abs_charge * pairc + rsum + 2. * qsum);
            for (int j = 0; j < N; ++j)
                mr[j] += main_counts[j * M + g] * gr;
        }
        for (int j = 0; j < I; ++j)
            mr[N + j] +=
                .5 * charge[j] * charge[j] * (deriv / avg + pd) + fabs(charge[j]) * pairc;
    }
    if (carbonate) {
        sr[co2] = lr[co2] = mr[co2] = 0.;
        for (int j = 0; j < I; ++j)
            mr[co2] += 2. * mol[j] * lambda_co2[j];
    }
    for (int j = 0; j < S; ++j)
        lng[j] = sr[j] + mr[j] + lr[j] - (j < N ? 0. : scale);
};
int evaluations = 0, retries = 0;
double errors[4] = {0., 0., 0., 0.};
