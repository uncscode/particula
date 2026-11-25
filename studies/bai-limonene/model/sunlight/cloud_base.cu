// Batched implicit parcel growth. One block shares one inventory and air state.
// The coupled saturation Newton update and warp reduction follow the reviewed
// citric-acid cloud parcel, with exact class-specific kappa-Kohler activities.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
constexpr double RD = 287.05799595894405, RV = 461.52280831345604;
constexpr double LV = 2.5e6, CP = 1005., WATER = 4188.790204786391;
__device__ double sat(double t) {
  return 611.2 * exp(LV / RV * (1. / 273.15 - 1. / t));
}
__device__ double ratio(double p, double t, double q) {
  return p * q / ((RD / RV + q) * sat(t));
}
__device__ double vol(double p, double t, double q) {
  return (RD + q * RV) * t / p;
}
__device__ double2 sum2(double a, double b) {
  __shared__ double2 warps[32];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int offset = 16; offset; offset /= 2) {
    a += __shfl_down_sync(0xffffffff, a, offset);
    b += __shfl_down_sync(0xffffffff, b, offset);
  }
  if (lane == 0)
    warps[warp] = make_double2(a, b);
  __syncthreads();
  if (warp == 0) {
    a = lane < blockDim.x / 32 ? warps[lane].x : 0;
    b = lane < blockDim.x / 32 ? warps[lane].y : 0;
    for (int offset = 16; offset; offset /= 2) {
      a += __shfl_down_sync(0xffffffff, a, offset);
      b += __shfl_down_sync(0xffffffff, b, offset);
    }
    if (lane == 0)
      warps[0] = make_double2(a, b);
  }
  __syncthreads();
  double2 out = warps[0];
  __syncthreads();
  return out;
}
__device__ double equilibrium(double u, double a3, double kelvin, double k,
                              double &deriv) {
  double r = sqrt(u), water = fmax(0., r * u - a3), den = water + k * a3;
  double ex = exp(kelvin / r), s = ex * water / den;
  deriv = ex * (1.5 * r) * (k * a3) / (den * den) - s * kelvin / (2. * u * r);
  return s;
}
__device__ double radius_step(double u0, double u, double dry, double a3,
                              double k, double kelvin, double gdt, double rh,
                              double &slope) {
  double lo = dry * dry, hi = u0 + gdt * rh;
  for (int iteration = 0; iteration < 30; ++iteration) {
    double ds;
    double s = equilibrium(u, a3, kelvin, k, ds);
    double residual = u - u0 - gdt * (rh - s);
    slope = gdt / fmax(.35, 1. + gdt * ds);
    if (fabs(residual) < 1.e-11 * u + 1e-26)
      break;
    if (residual > 0.)
      hi = u;
    else
      lo = u;
    double next = u - residual * slope / gdt;
    u = (next > lo && next < hi) ? next : .5 * (lo + hi);
  }
  return u;
}
// input columns: dry radius, count /kg, initial wet radius, kappa.
// initial air columns: p,T,q. velocity is supplied at each step.
__global__ __launch_bounds__(1024, 1) void integrate(
    const double *bins, const double *air, const double *vel, double *out,
    double *wet, double *traces, double *profiles, int n, int steps, double dt,
    int nh) {
  int i = threadIdx.x, b = blockIdx.x;
  const double *s = bins + 4 * (b * n + (i < n ? i : 0));
  double a = s[0], w = i < n ? s[1] : 0., rw = s[2], k = s[3], a3 = a * a * a,
         u = rw * rw;
  double p = air[3 * b], t = air[3 * b + 1], q = air[3 * b + 2];
  double l0 = sum2(WATER * w * (rw * u - a3), 0.).x, qt = q + l0;
  double peak = -1., werr = 0., implicit = 0., z = 0., hout = 0.;
  for (int h = 0; h < steps; h++) {
    double dz = dt * vel[b * steps + h];
    z += dz;
    double v = vol(p, t, q), dp = -(1 + qt) * 9.80665 * dz / v;
    p += dp;
    double td = t + v * dp / CP;
    double gdt =
        dt * 2. /
        (1000. *
         (RV * td / (2.11e-5 * pow(td / 273.15, 1.94) * 101325. / p * sat(td)) +
          LV / (.001 * (4.39 + .071 * td) * td) * (LV / (RV * td) - 1.)));
    double x = u, rh = ratio(p, td, q), dl = 0.;
    for (int j = 0; j < 8; j++) {
      double slope;
      x = radius_step(u, x, a, a3, k, .144 / (RV * 1000. * (td + LV / CP * dl)),
                      gdt, rh, slope);
      double r = sqrt(x);
      double2 s =
          sum2(w * WATER * (r * x - rw * u), w * WATER * 1.5 * r * slope);
      dl = s.x;
      double qn = q - dl, tn = td + LV / CP * dl, sr = ratio(p, tn, qn),
             err = rh - sr;
      if (fabs(err) < 2e-12 || j == 7) {
        implicit = fmax(implicit, fabs(err));
        break;
      }
      double sr_deriv = sr * ((RD / RV) / (qn * (RD / RV + qn)) +
                              LV * LV / (CP * RV * tn * tn));
      rh -= err / (1. + sr_deriv * s.y);
    }
    u = x;
    rw = sqrt(u);
    q -= dl;
    t = td + LV / CP * dl;
    double liquid = sum2(WATER * w * (rw * u - a3), 0.).x;
    werr = fmax(werr, fabs(q + liquid - qt));
    double ss = ratio(p, t, q) - 1.;
    if (ss > peak) {
      peak = ss;
      hout = z;
    }
    if (profiles && i < n && (h + 1) % (steps / nh) == 0)
      profiles[(b * nh + (h + 1) / (steps / nh) - 1) * n + i] = rw;
    if (i == 0 && (h + 1) % (steps / nh) == 0) {
      double *tr = traces + 5 * (b * nh + (h + 1) / (steps / nh) - 1);
      tr[0] = z;
      tr[1] = ss;
      tr[2] = liquid;
      tr[3] = t;
      tr[4] = p;
    }
  }
  if (i < n)
    wet[b * n + i] = rw;
  if (i == 0) {
    double *o = out + 8 * b;
    o[0] = p;
    o[1] = t;
    o[2] = q;
    o[3] = peak;
    o[4] = z;
    o[5] = werr;
    o[6] = implicit;
    o[7] = hout;
  }
}
int dispatch(const double *bins, const double *air, const double *vel,
             double *out, double *wet, double *trace, int b, int n, int steps,
             double dt, int nh, double *profile) {
  if (n > 1024 || steps % nh)
    return -1;
  size_t elements[] = {size_t(b) * n * 4, size_t(b) * 3, size_t(b) * steps,
                       size_t(b) * 8,     size_t(b) * n, size_t(b) * nh * 5};
  void *d[6] = {};
  const double *in[] = {bins, air, vel};
  double *dst[] = {out, wet, trace};
  cudaError_t e = cudaSuccess;
  for (int j = 0; j < 6 && !e; j++) {
    e = cudaMalloc(d + j, 8 * elements[j]);
    if (!e && j < 3)
      e = cudaMemcpy(d[j], in[j], 8 * elements[j], cudaMemcpyHostToDevice);
  }
  double *vertical = nullptr;
  if (!e && profile)
    e = cudaMalloc(&vertical, sizeof(double) * size_t(b) * nh * n);
  if (!e) {
    integrate<<<b, 32 * ((n + 31) / 32)>>>(
        (double *)d[0], (double *)d[1], (double *)d[2], (double *)d[3],
        (double *)d[4], (double *)d[5], vertical, n, steps, dt, nh);
    e = cudaGetLastError();
  }
  if (!e)
    e = cudaDeviceSynchronize();
  for (int j = 3; j < 6 && !e; j++)
    e = cudaMemcpy(dst[j - 3], d[j], 8 * elements[j], cudaMemcpyDeviceToHost);
  if (!e && profile)
    e = cudaMemcpy(profile, vertical, sizeof(double) * size_t(b) * nh * n,
                   cudaMemcpyDeviceToHost);
  if (vertical)
    cudaFree(vertical);
  for (auto p : d)
    if (p)
      cudaFree(p);
  if (e)
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
  return int(e);
}

extern "C" int evolve(const double *bins, const double *air, const double *vel,
                      double *out, double *wet, double *trace, int b, int n,
                      int steps, double dt, int nh) {
  return dispatch(bins, air, vel, out, wet, trace, b, n, steps, dt, nh,
                  nullptr);
}
extern "C" int evolve_profiles(const double *bins, const double *air,
                               const double *vel, double *out, double *wet,
                               double *trace, int b, int n, int steps,
                               double dt, int nh, double *profile) {
  return dispatch(bins, air, vel, out, wet, trace, b, n, steps, dt, nh,
                  profile);
}
