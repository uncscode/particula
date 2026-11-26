// Size-dependent photolysis and simultaneous conservative gas-particle
// exchange. One OpenMP thread owns a 72-hour independent cohort and its vapor
// reservoir.
#include <algorithm>
#include <cmath>
#include <vector>
constexpr int S = 15;
constexpr double pi = 3.141592653589793;
extern "C" void
seasonal_age(int boxes, int n, int hours, int sub, int grid, const double *flux,
             const double *original, const double *number, const double *core,
             const double *mass, const double *corek, const double *absorb_core,
             const double *init, const double *C, const double *k,
             const double *lut, const double *lensf, const double *lenss,
             double jf, double js, double persistence, double *diam,
             double *kapp, double *audit) {
  const double dt = 3600. / sub, sink = dt / 86400.,
               speed = std::sqrt(8 * 8.314 * 294 / (pi * .25));
#pragma omp parallel for schedule(dynamic)
  for (int b = 0; b < boxes; b++) {
    std::vector<double> M(init, init + n * S), inv(n * S), drive(n);
    double gas[S] = {}, escaped = 0, exported = 0, reacted2 = 0, initial = 0;
    for (double v : M)
      initial += v;
    for (int t = 0; t < hours * sub; t++) {
      const int hour = t / sub;
      // Even-duration windows with two half endpoints have an odd number of records.
      if (hours % 2 == 1 && ((hour == 0 && t % sub < sub / 2) ||
                          (hour == hours - 1 && t % sub >= sub / 2)))
        continue;
      const double *f = flux + (b * hours + hour) * 2;
      double num[S], den[S];
      for (int s = 0; s < S; s++) {
        num[s] = gas[s];
        den[s] = 1 + sink;
      }
      for (int i = 0; i < n; i++) {
        double *m = &M[i * S];
        double dry = core[i];
        for (int s = 0; s < S; s++)
          dry += m[s];
        double d = original[i] * std::cbrt(dry / mass[i]);
        if (f[0] > 0 || f[1] > 0) {
          double x = std::log(d);
          int j = std::clamp(
              int(std::upper_bound(lut, lut + grid, x) - lut) - 1, 0, grid - 2);
          double u = std::clamp((x - lut[j]) / (lut[j + 1] - lut[j]), 0., 1.);
          double ls =
              js * f[1] * (lenss[j] + u * (lenss[j + 1] - lenss[j])) * dt;
          double slow = -std::expm1(-ls);
          double fast = -std::expm1(
              -jf * f[0] * (lensf[j] + u * (lensf[j + 1] - lensf[j])) * dt);
          double second = -std::expm1(-persistence * ls);
          double ssum = 0, fsum = 0;
          for (int s = 0; s < 5; s++) {
            double v = m[s] * slow;
            ssum += v;
            m[s] -= v;
          }
          for (int s = 5; s < 10; s++) {
            double v = m[s] * fast;
            fsum += v;
            m[s] -= v;
          }
          for (int s = 10; s < S; s++) {
            double v = m[s] * second;
            reacted2 += v;
            escaped += v;
            m[s] -= v;
          }
          m[10] += .40 * ssum;
          for (int s = 11; s < S; s++)
            m[s] += (.41 * ssum + .68 * fsum) / 4;
          escaped += .19 * ssum + .32 * fsum;
        }
        double absorb = absorb_core[i], ksum = core[i] * corek[i];
        for (int s = 0; s < S; s++) {
          absorb += m[s];
          ksum += m[s] * k[s];
        }
        double r = std::max(d * 5e-10 * std::cbrt(1 + 4 * ksum / dry), 1e-10);
        double kn = 3 * 8e-6 / (speed * r);
        double beta = (1 + kn) / (1 + (4. / 3 + .377) * kn + 4. / 3 * kn * kn);
        drive[i] = dt * 4 * pi * 8e-6 * r * beta * number[i] * 1e6;
        double a = drive[i] *
                   std::exp(std::min(25., .015 / (8.314 * 294 * 1250 * r))) /
                   std::max(1e-25, absorb);
        for (int s = 0; s < S; s++) {
          double v = 1 / (1 + a * C[s]);
          inv[i * S + s] = v;
          num[s] += m[s] * (1 - v);
          den[s] += drive[i] * v;
        }
      }
      for (int s = 0; s < S; s++) {
        gas[s] = num[s] / den[s];
        exported += sink * gas[s];
      }
      for (int i = 0; i < n; i++)
        for (int s = 0; s < S; s++)
          M[i * S + s] = (M[i * S + s] + drive[i] * gas[s]) * inv[i * S + s];
    }
    double condensed = 0, ambient = 0;
    for (int s = 0; s < S; s++)
      ambient += gas[s];
    for (int i = 0; i < n; i++) {
      double d = core[i], kk = core[i] * corek[i];
      for (int s = 0; s < S; s++) {
        double v = M[i * S + s];
        condensed += v;
        d += v;
        kk += v * k[s];
      }
      diam[b * n + i] = original[i] * std::cbrt(d / mass[i]);
      kapp[b * n + i] = kk / d;
    }
    double *a = audit + b * 5;
    a[0] = condensed / initial;
    a[1] = escaped / initial;
    a[2] = reacted2 / initial;
    a[3] = exported / initial;
    a[4] =
        std::abs(initial - condensed - escaped - exported - ambient) / initial;
  }
}
