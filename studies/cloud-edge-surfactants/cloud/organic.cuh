// Reversible exchange of one externally supplied, water-soluble organic vapor.
// SI units throughout. Each mass or mole count is for ONE physical particle.
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define ORG_INLINE __host__ __device__ __forceinline__
#else
#define ORG_INLINE inline
#endif

// Pure-component saturation vapor mass concentration [kg/m3]. The factor
// tref/T converts the Clausius-Clapeyron vapor-pressure ratio into concentrations.
ORG_INLINE double org_saturation(double temperature, double cstar_ref,
                                 double tref, double dh_molar, double /*mw*/) {
    constexpr double gas_r = 8.314462618;
    return cstar_ref * (tref / temperature) *
        exp((dh_molar / gas_r) * (1.0 / tref - 1.0 / temperature));
}

// Kelvin correction from the organic partial molar volume, approximated by
// pure organic molar volume. Use the mixture surface tension in N/m.
ORG_INLINE double org_kelvin(double wet_radius, double temperature,
                            double sigma, double mw, double organic_density) {
    constexpr double gas_r = 8.314462618;
    return exp(2.0 * sigma * mw /
               (wet_radius * gas_r * temperature * organic_density));
}

// Dref [m2/s] is at 298.15 K and 101325 Pa. Fuchs-Sutugin transition factor
// depends on organic mean speed and vapor diffusivity, not air mean free path.
ORG_INLINE double org_conductance(double wet_radius, double temperature,
                                 double pressure, double dref, double alpha,
                                 double mw) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double gas_r = 8.314462618;
    const double diffusion = dref * pow(temperature / 298.15, 1.75) * (101325.0 / pressure);
    const double speed = sqrt(8.0 * gas_r * temperature / (pi * mw));
    const double kn = 3.0 * diffusion / (wet_radius * speed);
    const double a = 4.0 / (3.0 * alpha);
    const double transition = (1.0 + kn) / (1.0 + (0.377 + a) * kn + a * kn * kn);
    return 4.0 * pi * wet_radius * diffusion * transition;
}

// Positive mass solution of backward Euler exchange. Equilibrium surface vapor
// concentration = cstar_kelvin * m / (m + n_abs * mw). Caller supplies a
// nonnegative shared gas concentration and positive background absorbing moles.
// With dt=0 the returned mass equals the initial mass exactly.
ORG_INLINE double org_target_mass(double m_old, double cgas, double dt,
                                 double conductance, double cstar_kelvin,
                                 double n_abs, double mw) {
    if (dt == 0.0) return m_old;
    const double h = dt * conductance;
    const double bmass = n_abs * mw;
    const double source = m_old + h * cgas;
    if (source == 0.0) return 0.0;
    const double b = bmass + h * cstar_kelvin - source;
    const double d = sqrt(b * b + 4.0 * bmass * source);
    return b >= 0.0 ? 2.0 * bmass * source / (d + b) : (d - b) * 0.5;
}

ORG_INLINE double org_absorber_moles(double water_moles, double acid_moles,
                                    double accessible_water) {
    return acid_moles + accessible_water * water_moles;
}

// Incremental osmotic effect relative to the recipe's background activity.
// The effective pre-organic mole denominator is water_moles / aw_background.
// For a neutral, fully dissolved organic set nu=1; nu=0 is an isolation control.
ORG_INLINE double org_water_activity(double aw_background, double water_moles,
                                    double org_moles, double nu) {
    if (org_moles == 0.0 || nu == 0.0 || aw_background == 0.0) return aw_background;
    return aw_background * water_moles / (water_moles + nu * org_moles * aw_background);
}

#undef ORG_INLINE
