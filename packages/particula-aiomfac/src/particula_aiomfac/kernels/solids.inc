// Crystal amounts are derived afresh from the particle's total ion inventory.
// The charge-neutral reaction columns may draw on ions from different source
// salts. Every trial liquid internally equilibrates its aqueous reactions.
const int mineral_count = minerals.shape[0];
double *free_moles = mineral_work.data + row * mineral_work.shape[1];
double *base = free_moles + D, *liquid_mass = base + D, *extent = liquid_mass + D;
double max_mass = 0., solvent_mass = 0.;
for (int k = 0; k < D; ++k) {
    if (!isfinite(mass[k]) || mass[k] < 0.) {
        status.data[row] = -1;
        return;
    }
    max_mass = fmax(max_mass, mass[k]);
}
if (!(max_mass > 0.)) {
    status.data[row] = -1;
    return;
}
for (int k = 0; k < D; ++k) {
    free_moles[k] = (mass[k] / max_mass) / formula_mw.data[k];
    if (k < N)
        solvent_mass += mass[k] / max_mass;
}
if (!(solvent_mass > 0.)) {
    status.data[row] = -20;
    return;
}
for (int s = 0; s < mineral_count; ++s) {
    const double *spec = minerals.data + s * minerals.shape[1];
    if (!(T >= spec[1] && T <= spec[2])) {
        status.data[row] = -21;
        return;
    }
    extent[s] = 0.;
}
const bool reactive_carbon = carbonate && gibbs_carbonate;
if (reactive_carbon) {
    const double carbon_salts = free_moles[co] + free_moles[hc];
    free_moles[water] += carbon_salts + free_moles[oh];
    free_moles[co2] += carbon_salts;
    free_moles[co] = free_moles[hc] = free_moles[oh] = free_moles[h] = 0.;
}
auto saturation = [&](int crystal) {
    const double *spec = minerals.data + crystal * minerals.shape[1];
    double si = -spec[0];
    for (int i = 0; i < I; ++i)
        if (spec[3 + i] > 0.) {
            if (out_mol[i] == 0.)
                return -1.e300;
            si += spec[3 + i] * (log(out_mol[i]) + out_ion_log[i]) / log(10.);
        }
    return si;
};
auto evaluate_free = [&](bool stable) __attribute__((noinline)) {
    for (int k = 0; k < D; ++k)
        liquid_mass[k] = free_moles[k] * formula_mw.data[k];
    if (reactive_carbon) {
        // Reconstruct charge-balanced aqueous trial amounts from conserved
        // carbon, water, and spectator ion totals after removing each crystal.
        // The inner mass-action solve determines actual chemical species.
        double net = 0., absolute = 0.;
        for (int k = 0; k < I; ++k) {
            const double amount = free_moles[N + k] * charge[k];
            net += amount;
            absolute += fabs(amount);
        }
        if (fabs(net) < 8. * 2.220446049250313e-16 * absolute)
            net = 0.;
        const double carbon = free_moles[co2];
        double solvent_water = free_moles[water];
        if (net > 0.) {
            // CO2 and OH- carry independent carbon and alkalinity.
            liquid_mass[oh] = net * formula_mw.data[oh];
            solvent_water -= net;
        } else if (carbon > 0. && absolute > 0.) {
            // H+ and CO3-- carry independent carbon and acidity.
            solvent_water -= carbon;
            liquid_mass[co2] = 0.;
            liquid_mass[co] = carbon * formula_mw.data[co];
            liquid_mass[h] = (2. * carbon - net) * formula_mw.data[h];
        } else
            liquid_mass[h] = -net * formula_mw.data[h];
        if (solvent_water <= 0.) {
            status.data[row] = -24;
            return false;
        }
        liquid_mass[water] = solvent_water * formula_mw.data[water];
    }
    if (stable)
        liquid(liquid_mass);
    else
        chemistry(liquid_mass);
    return status.data[row] >= 0;
};
auto complementarity = [&]() {
    double error = 0.;
    for (int s = 0; s < mineral_count; ++s) {
        const double si = saturation(s);
        if (!isfinite(si))
            return 1.e300;
        error = fmax(error, extent[s] > 0. ? fabs(si) : fmax(0., si));
    }
    return error;
};
// First dissolve/precipitate on the homogeneous branch; this bounds ionic
// strength before evaluating candidate liquid splits in a nearly dry particle.
// Only a solution passing complementarity on stable liquids is accepted.
for (int stage = 0; stage < 2; ++stage) {
    bool stable = stage == 1;
    if (stable && evaluate_free(true) && complementarity() <= 1.e-7)
        return;
    for (int sweep = 0; sweep < 100; ++sweep) {
        for (int s = 0; s < mineral_count; ++s) {
            const double *nu = minerals.data + s * minerals.shape[1] + 3 + I;
            double maximum = 1.e300;
            for (int i = 0; i < D; ++i) {
                base[i] = free_moles[i] + extent[s] * nu[i];
                if (nu[i] > 0.)
                    maximum = fmin(maximum, base[i] / nu[i]);
            }
            if (maximum <= 0.)
                continue;
            for (int i = 0; i < D; ++i) {
                const double available = base[i];
                const double excess = available - maximum * nu[i];
                // Different mass-to-mole conversions reconstruct the *same*
                // binary salt up to floating-point roundoff. Avoid treating
                // that cancellation as a real counterion in a dry film.
                base[i] = excess > 8. * 2.220446049250313e-16 * available ? excess : 0.;
            }
            auto trial = [&](double dissolved) {
                for (int i = 0; i < D; ++i)
                    free_moles[i] = base[i] + dissolved * nu[i];
                if (!evaluate_free(stable))
                    return double(NAN);
                return saturation(s);
            };
            // Search outward from dilute amounts before entering concentrated
            // regions where the metastable activity may be nonmonotonic.
            double lo = 0., hi = fmin(maximum, solvent_mass), value = trial(hi);
            while (value < 0. && hi < maximum) {
                lo = hi;
                hi = fmin(maximum, hi * 2.);
                value = trial(hi);
            }
            if (!isfinite(value)) {
                status.data[row] = -22;
                return;
            }
            double dissolved = hi;
            if (value > 0.) {
                for (int iter = 0; iter < 120; ++iter) {
                    dissolved = .5 * (lo + hi);
                    value = trial(dissolved);
                    if (!isfinite(value)) {
                        status.data[row] = -22;
                        return;
                    }
                    if (fabs(value) < 1.e-10)
                        break;
                    if (value > 0.)
                        hi = dissolved;
                    else
                        lo = dissolved;
                }
            }
            extent[s] = maximum - dissolved;
        }
        if (!evaluate_free(stable))
            return;
        if (complementarity() <= 1.e-7) {
            if (stable)
                return;
            break;
        }
        if (sweep == 99) {
            status.data[row] = -23;
            return;
        }
    }
}
status.data[row] = -23;
