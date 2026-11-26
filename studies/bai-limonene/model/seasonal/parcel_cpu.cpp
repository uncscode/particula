// Independent host replay of the private parcel reference, with no thread-block
// limit on the number of weighted dry sizes. Parallelism is across parcels.
#include <algorithm>
#include <cmath>
#include <vector>
constexpr double rd = 287.05799595894405, rv = 461.52280831345604, lv = 2.5e6,
                 cp = 1005., water = 4188.790204786391;
double saturation(double t) {
  return 611.2 * std::exp(lv / rv * (1. / 273.15 - 1. / t));
}
double rh_air(double p, double t, double q) {
  return p * q / ((rd / rv + q) * saturation(t));
}
double step_radius(double u0, double u, double dry, double a3, double k,
                   double kelvin, double gdt, double rh, double &slope) {
  double lo = dry * dry, hi = u0 + gdt * rh;
  for (int j = 0; j < 30; j++) {
    double r = std::sqrt(u), v = std::max(0., r * u - a3), den = v + k * a3;
    double ex = std::exp(kelvin / r), se = ex * v / den;
    double deriv =
        ex * 1.5 * r * k * a3 / (den * den) - se * kelvin / (2 * u * r);
    double error = u - u0 - gdt * (rh - se);
    slope = gdt / std::max(.35, 1 + gdt * deriv);
    if (std::abs(error) < 1e-11 * u + 1e-26)
      break;
    if (error > 0)
      hi = u;
    else
      lo = u;
    double next = u - error * slope / gdt;
    u = next > lo && next < hi ? next : .5 * (lo + hi);
  }
  return u;
}
extern "C" int evolve_profiles(const double *bins, const double *air,
                               const double *vel, double *out, double *wet,
                               double *trace, int boxes, int n, int steps,
                               double dt, int nh, double *profiles) {
  if (steps % nh)
    return -1;
#pragma omp parallel for schedule(dynamic)
  for (int b = 0; b < boxes; b++) {
    auto row = bins + 4 * b * n;
    double p = air[3 * b], t = air[3 * b + 1], q = air[3 * b + 2];
    std::vector<double> dry(n), k(n), num(n), a3(n), u(n), rw(n), x(n), rnew(n);
    double qt = q;
    for (int i = 0; i < n; i++) {
      dry[i] = row[4 * i];
      num[i] = row[4 * i + 1] * water;
      rw[i] = row[4 * i + 2];
      k[i] = row[4 * i + 3];
      a3[i] = dry[i] * dry[i] * dry[i];
      u[i] = rw[i] * rw[i];
      qt += num[i] * (rw[i] * u[i] - a3[i]);
    }
    double peak = -1, werr = 0, implicit = 0, z = 0, zpeak = 0;
    for (int h = 0; h < steps; h++) {
      double dz = dt * vel[b * steps + h];
      z += dz;
      double v = (rd + q * rv) * t / p, dp = -(1 + qt) * 9.80665 * dz / v;
      p += dp;
      double td = t + v * dp / cp;
      double gdt = dt * 2 /
                   (1000 * (rv * td /
                                (2.11e-5 * std::pow(td / 273.15, 1.94) *
                                 101325. / p * saturation(td)) +
                            lv / (.001 * (4.39 + .071 * td) * td) *
                                (lv / (rv * td) - 1)));
      double rh = rh_air(p, td, q), dl = 0;
      x = u;
      for (int pass = 0; pass < 8; pass++) {
        double kelvin = .144 / (rv * 1000 * (td + lv / cp * dl));
        double gain = 0, deriv = 0;
        for (int i = 0; i < n; i++) {
          double slope;
          x[i] = step_radius(u[i], x[i], dry[i], a3[i], k[i], kelvin, gdt, rh,
                             slope);
          rnew[i] = std::sqrt(x[i]);
          gain += num[i] * (rnew[i] * x[i] - rw[i] * u[i]);
          deriv += num[i] * 1.5 * rnew[i] * slope;
        }
        dl = gain;
        double qn = q - dl, tn = td + lv / cp * dl, sr = rh_air(p, tn, qn),
               error = rh - sr;
        if (std::abs(error) < 2e-12 || pass == 7) {
          implicit = std::max(implicit, std::abs(error));
          break;
        }
        double srd = sr * ((rd / rv) / (qn * (rd / rv + qn)) +
                           lv * lv / (cp * rv * tn * tn));
        rh -= error / (1 + srd * deriv);
      }
      u.swap(x);
      rw.swap(rnew);
      q -= dl;
      t = td + lv / cp * dl;
      double liq = 0;
      for (int i = 0; i < n; i++)
        liq += num[i] * (rw[i] * u[i] - a3[i]);
      werr = std::max(werr, std::abs(q + liq - qt));
      double ss = rh_air(p, t, q) - 1;
      if (ss > peak) {
        peak = ss;
        zpeak = z;
      }
      if ((h + 1) % (steps / nh) == 0) {
        int l = (h + 1) / (steps / nh) - 1;
        double *tr = trace + (b * nh + l) * 5;
        tr[0] = z;
        tr[1] = ss;
        tr[2] = liq;
        tr[3] = t;
        tr[4] = p;
        if (profiles)
          std::copy(rw.begin(), rw.end(), profiles + (b * nh + l) * n);
      }
    }
    std::copy(rw.begin(), rw.end(), wet + b * n);
    double *dest = out + 8 * b;
    double values[] = {p, t, q, peak, z, werr, implicit, zpeak};
    std::copy(values, values + 8, dest);
  }
  return 0;
}

// Passive tracer intervention on a previously integrated air history. Matched
// histories recover the coupled trajectory; cross histories are diagnostic.
extern "C" void replay_history(const double *bins,const double *air,
                               const double *trace,double *final_radii,
                               int boxes,int n,int steps,double dt) {
  #pragma omp parallel for schedule(dynamic)
  for (int b=0;b<boxes;b++) {
    const double *row=bins+4*b*n;
    double p=air[3*b], t=air[3*b+1], q=air[3*b+2];
    std::vector<double> u(n),a3(n);
    for(int i=0;i<n;i++) {
      u[i]=row[4*i+2]*row[4*i+2];
      a3[i]=row[4*i]*row[4*i]*row[4*i];
    }
    for(int h=0;h<steps;h++) {
      const double *tr=trace+(b*steps+h)*5;
      const double next_p=tr[4], next_t=tr[3], rh=1+tr[1];
      const double td=t+(rd+q*rv)*t/p*(next_p-p)/cp;
      const double gdt=dt*2/(1000*(rv*td/(2.11e-5*std::pow(td/273.15,1.94)*101325./next_p*saturation(td))
                          +lv/(.001*(4.39+.071*td)*td)*(lv/(rv*td)-1)));
      const double kelvin=.144/(rv*1000*next_t);
      for(int i=0;i<n;i++) {
        double slope;
        u[i]=step_radius(u[i],u[i],row[4*i],a3[i],row[4*i+3],kelvin,gdt,rh,slope);
      }
      p=next_p;t=next_t;double e=rh*saturation(t);q=rd/rv*e/(p-e);
    }
    for(int i=0;i<n;i++)final_radii[b*n+i]=std::sqrt(u[i]);
  }
}
