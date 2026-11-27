// Two-reservoir propagation with exact phase transfer and first-order vapor
// loss over a frozen-radius substep. Particle and gas inventories are per
// species, in units of that particle's initial dry mass (ambient gas divided by
// number).
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

extern "C" void integrate(const double *weights, const double *times, int nt,
                          double m0, double r0, double mfp, double diffusivity,
                          double kelvin_length, double number, double tau,
                          double step_max, double speed, double *out) {
  std::array<double, 7> particle, gas{};
  std::copy_n(weights, 7, particle.begin());
  double time = 0.;
  for (int j = 0; j < nt; ++j) {
    const double until = times[j] * 60.;
    while (time < until) {
      const double dt = std::min({until - time, step_max, 0.03 * (time + 1.)});
      double total = 0.;
      for (double v : particle)
        total += v;
      const double r = r0 * std::cbrt(total);
      if (r <= .5e-9) {
        std::fill(out + j, out + nt, 0.);
        return;
      }
      const double kn = mfp / r;
      const double correction = .75 * (1. + kn) / (kn * kn + 1.283 * kn + .75);
      const double k =
          4. * std::numbers::pi * r * diffusivity * correction * speed;
      const double back = k * number;
      const double leak = 1. / tau;
      const double unit_loss =
          k * 1e-9 * std::exp(kelvin_length / r) / (m0 * total);
      double cstar = 1e-4;
      for (int i = 0; i < 7; ++i) {
        const double loss = unit_loss * cstar;
        const double sum = loss + back + leak;
        const double gap = std::sqrt(
            (loss - back - leak) * (loss - back - leak) + 4. * loss * back);
        const double fast = .5 * (sum + gap);
        const double slow = loss * leak / fast;
        const double a = std::exp(-slow * dt);
        const double b = std::exp(-fast * dt);
        const double gain = (a - b) / gap;
        const double exchange = loss * particle[i] - back * gas[i];
        const double p_next =
            b * particle[i] + gain * (fast * particle[i] - exchange);
        const double g_next =
            b * gas[i] + gain * ((fast - leak) * gas[i] + exchange);
        particle[i] = p_next;
        gas[i] = g_next;
        cstar *= 10.;
      }
      time += dt;
    }
    out[j] = 0.;
    for (double v : particle)
      out[j] += v;
  }
}
