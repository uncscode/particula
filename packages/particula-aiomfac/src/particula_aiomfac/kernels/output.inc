terms();
auto safe_exp = [&](double y, double fraction) {
    double lim = fraction * .49 * 709.782712893384;
    return exp(copysign(
        fmin(fabs(y), lim) + fmin(.01 * log1p(fmax(0., fabs(y) - lim)), .04 * lim), y));
};
for (int j = 0; j < S; ++j) {
    if (!isfinite(lng[j]) || !isfinite(amounts[j])) {
        fail(4);
        return;
    }
    out_sr[j] = sr[j];
    out_mr[j] = mr[j];
    out_lr[j] = lr[j];
    out_moles[j] = amounts[j];
    if (j < N) {
        bool defined = amounts[j] * mw[j] > 1.4916681462400413e-154;
        out_x[j] = x[j];
        out_neutral_mask[j] = defined;
        out_neutral_log[j] = defined ? lng[j] : NAN;
        out_activity[j] = defined ? x[j] * safe_exp(lng[j], .4) : 0.;
    } else {
        int i = j - N;
        bool defined = mol[i] > 1.4916681462400413e-154;
        out_mol[i] = mol[i];
        out_ion_mask[i] = defined;
        out_ion_log[i] = defined ? lng[j] : NAN;
    }
}
for (int c = 0; c < C; ++c)
    for (int a = 0; a < A; ++a) {
        int k = c * A + a;
        bool defined = out_ion_mask[c] && out_ion_mask[C + a];
        double vc = pair_counts[2 * k], va = pair_counts[2 * k + 1];
        double v = vc * lng[N + c] + va * lng[N + C + a];
        out_pair_mask[k] = defined;
        out_mean[k] = defined ? safe_exp(v / (vc + va), 1.) : NAN;
        out_iap[k] =
            defined ? safe_exp(v + vc * log(mol[c]) + va * log(mol[C + a]), 1.) : 0.;
    }
out_solid[0] = solid;
out_strength[0] = strength;
out_scale[0] = scale;
out_warning[0] =
    I ? ((T < 288. || T > 309.) ? 10 : 16) : ((T < 280. || T > 400.) ? 11 : 16);
out_evaluations[0] = evaluations;
out_retries[0] = retries;
for (int k = 0; k < 4; ++k)
    out_errors[k] = errors[k];
status.data[row] = evaluations ? 1 : 0;
