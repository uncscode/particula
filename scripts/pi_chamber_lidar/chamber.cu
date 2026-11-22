// A block advances one chamber. Each warp holds one well-mixed air volume.
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

constexpr int WARP = 32, MAX_BINS = 212, MAX_ZONES = 12;
constexpr double SAT =
    2338.3 / ((8.314462618 / .01801528) * 293.15) * 1e6; // mg/m3
constexpr double QSEED = .002 / (60. * 3.141592653589793);
constexpr double QENT = .020 / (60. * 3.141592653589793);
enum InletLayout { LAYER_INLET, PATCH_INLET, UNIFORM_SEED_REFERENCE };
enum Budget {
  FEED,
  FLOOR_N,
  WALL_N,
  VENT_N,
  WATER_IN,
  VENT_W,
  FLOOR_W,
  WALL_W,
  COND,
  EVAP,
  ENTER_VISIBLE,
  LEAVE_VISIBLE,
  RHMIN,
  RHMAX,
  NBUD
};

__device__ double warp_sum(double v) {
  for (int j = 16; j; j >>= 1)
    v += __shfl_down_sync(0xffffffff, v, j);
  return __shfl_sync(0xffffffff, v, 0);
}

__device__ unsigned mix_bits(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  return x ^ (x >> 16);
}

// Air and concentration anomalies decay exactly under symmetric exchange.
__device__ double exchange(const double *x, int box, int stride, int k,
                           int layers, int patches, double vertical,
                           double local, double mode1, double mode3) {
  int layer = box / patches, patch = box % patches;
  if (layers == 3) {
    // Exact reflecting, nearest-neighbor diffusion on three equal volumes.
    const double odd[3] = {1., 0., -1.}, even[3] = {1., -2., 1.};
    double mixed = 0.;
    for (int z = 0; z < 3; z++) {
      double mean = 0.;
      for (int j = 0; j < patches; j++)
        mean += x[(z * patches + j) * stride + k];
      mean /= patches;
      double weight = 1. / 3. + .5 * mode1 * odd[layer] * odd[z] +
                      mode3 * even[layer] * even[z] / 6.;
      mixed += weight *
               (mean + local * (x[(z * patches + patch) * stride + k] - mean));
    }
    return mixed;
  }
  double own = 0, other = 0;
  for (int j = 0; j < patches; j++) {
    own += x[(layer * patches + j) * stride + k];
    other += x[((layers - 1 - layer) * patches + j) * stride + k];
  }
  own /= patches;
  other /= patches;
  double a = own + local * (x[(layer * patches + patch) * stride + k] - own);
  double b = other +
             local * (x[((layers - 1 - layer) * patches + patch) * stride + k] -
                      other);
  return a + vertical * (b - a);
}

__global__ void advance(int n, int h, int layers, int patches, int intermittent,
                        int inlet_layout, int entrain, int steps, int offset,
                        double dt, double inlet, int seedbin, double seedweight,
                        const double *rh, const double *mapmass,
                        const int *dest, const double *weight,
                        const double *mass, const double *speed,
                        const double *diam, const double *pars, double *conc,
                        double *gas, double *sums, double *integrals) {
  __shared__ double a[MAX_ZONES * MAX_BINS], b[MAX_ZONES * MAX_BINS],
      g[MAX_ZONES];
  int caseid = blockIdx.x, box = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  int zones = layers * patches, layer = box / patches, patch = box % patches;
  const double *p = pars + caseid * 9;
  double *counts = conc + (caseid * zones + box) * n;
  double *bod = sums + (caseid * zones + box) * NBUD;
  double *avg = integrals + (caseid * zones + box) * n;
  double vs = .5 * (-expm1(-dt / p[1])), ls = exp(-dt / p[4]);
  double mode1 = exp(-1.125 * dt / p[1]), mode3 = mode1 * mode1 * mode1;
  for (int k = lane; k < n; k += WARP)
    a[box * n + k] = counts[k];
  if (!lane)
    g[box] = gas[caseid * zones + box];
  __syncthreads();
  for (int step = 0; step < steps; step++) {
    unsigned epoch = (unsigned)floor(((offset + step + .5) * dt) / p[5]);
    unsigned key = (unsigned)p[7] + epoch * 0x9e3779b9U;
    int wet_id = mix_bits(key + (layer + 1) * 0xa511e9b3U) % patches;
    int dry_id = mix_bits(key + 0x74b12d29U) % patches;
    double q = p[0] * (layers == 1 ? 1. : 2. * (layer == 0 ? p[3] : 1. - p[3]));
    if (layers == 3 && layer == 1)
      q = p[0];
    if (intermittent)
      q *= (1. - p[6]) + p[6] * patches * (patch == wet_id);
    double dryrate = entrain && layer == layers - 1 ? QENT * layers : 0.;
    if (inlet_layout == PATCH_INLET)
      dryrate *= patches * (patch == dry_id);
    double source_factor =
        (layers == 3 && inlet_layout != UNIFORM_SEED_REFERENCE)
            ? 3. * (layer == 1)
            : 1.;
    double carrier = source_factor * QSEED;
    double air = carrier + dryrate, exhausted = -expm1(-dt * air);
    double oldgas =
        exchange(g, box, 1, 0, layers, patches, vs, ls, mode1, mode3);
    double supplied = dt * q + exhausted * (carrier * .95 + dryrate * inlet) /
                                   fmax(air, 1.e-30) * SAT;
    double nfeed = source_factor * p[8] * dt;
    double seedmass =
        mass[seedbin] * (1. - seedweight) + mass[seedbin + 1] * seedweight;
    double available = oldgas * (1. - exhausted) + supplied - nfeed * seedmass;
    // Inject hydrated seeds and debit their water from the gas.
    for (int k = lane; k < n; k += WARP)
      b[box * n + k] =
          exchange(a, box, n, k, layers, patches, vs, ls, mode1, mode3);
    __syncthreads();
    double floorN = 0, wallN = 0, ventN = 0, floorW = 0, wallW = 0, ventW = 0;
    for (int k = lane; k < n; k += WARP) {
      double original = b[box * n + k];
      double moved = -expm1(-dt * layers * speed[k]) * original;
      double left = original - moved;
      if (layer + 1 < layers)
        left += -expm1(-dt * layers * speed[k]) * b[(box + patches) * n + k];
      if (layer == 0) {
        floorN += moved;
        floorW += moved * mass[k];
      }
      double wall = diam[k] < 2. ? -expm1(-dt / p[2]) * left : 0.;
      double vent = (left - wall) * exhausted;
      wallN += wall;
      wallW += wall * mass[k];
      ventN += vent;
      ventW += vent * mass[k];
      a[box * n + k] = left - wall - vent +
                       nfeed * ((k == seedbin) * (1. - seedweight) +
                                (k == seedbin + 1) * seedweight);
    }
    __syncthreads();
    // Solve the local implicit vapor balance. The transfer table advances
    // each wet pivot for dt at constant RH; interpolate its two remaps.
    int lo = 0, hi = h - 1;
    while (hi - lo > 1) {
      int mid = (lo + hi) / 2;
      double take = 0;
      for (int k = lane; k < n; k += WARP)
        take += a[box * n + k] * (mapmass[mid * n + k] - mass[k]);
      if (SAT * rh[mid] + warp_sum(take) > available)
        hi = mid;
      else
        lo = mid;
    }
    double take0 = 0, take1 = 0;
    for (int k = lane; k < n; k += WARP) {
      double num = a[box * n + k];
      take0 += num * (mapmass[lo * n + k] - mass[k]);
      take1 += num * (mapmass[hi * n + k] - mass[k]);
      b[box * n + k] = 0;
    }
    double left = SAT * rh[lo] + warp_sum(take0),
           right = SAT * rh[hi] + warp_sum(take1);
    double t = (available - left) / (right - left);
    if (!(t >= -1e-8 && t <= 1. + 1e-8))
      t = NAN; // reported domain failure
    t = fmin(1., fmax(0., t)) + (isnan(t) ? NAN : 0.);
    double actual_rh = rh[lo] + t * (rh[hi] - rh[lo]);
    __syncwarp();
    double cond = 0, evap = 0, up = 0, down = 0;
    for (int k = lane; k < n; k += WARP) {
      double num = a[box * n + k];
      bool before = diam[k] >= .583 && diam[k] <= 40.679;
      for (int j = 0; j < 2; j++) {
        int index = (j ? hi : lo) * n + k, i = dest[index];
        double count = num * (j ? t : 1. - t), w = weight[index];
        atomicAdd(&b[box * n + i], count * (1. - w));
        atomicAdd(&b[box * n + i + 1], count * w);
        double flux = count * (mapmass[index] - mass[k]);
        cond += fmax(flux, 0.);
        evap += fmax(-flux, 0.);
        double seen = (1. - w) * (diam[i] >= .583 && diam[i] <= 40.679) +
                      w * (diam[i + 1] >= .583 && diam[i + 1] <= 40.679);
        up += before ? 0. : count * seen;
        down += before ? count * (1. - seen) : 0.;
      }
    }
    double increment[12] = {nfeed,
                            warp_sum(floorN),
                            warp_sum(wallN),
                            warp_sum(ventN),
                            supplied,
                            exhausted * oldgas + warp_sum(ventW),
                            warp_sum(floorW),
                            warp_sum(wallW),
                            warp_sum(cond),
                            warp_sum(evap),
                            warp_sum(up),
                            warp_sum(down)};
    if (!lane) {
      g[box] = SAT * actual_rh;
      for (int i = 0; i < 12; i++)
        bod[i] += increment[i];
      bod[RHMIN] = fmin(bod[RHMIN], actual_rh);
      bod[RHMAX] = fmax(bod[RHMAX], actual_rh);
    }
    __syncthreads();
    for (int k = lane; k < n; k += WARP) {
      a[box * n + k] = b[box * n + k];
      avg[k] += dt * a[box * n + k];
    }
    __syncthreads();
  }
  for (int k = lane; k < n; k += WARP)
    counts[k] = a[box * n + k];
  if (!lane)
    gas[caseid * zones + box] = g[box];
}

// Python owns device arrays (Warp); counters and checkpoints persist on device.
extern "C" int launch(int cases, int n, int h, int layers, int patches,
                      int pulse, int inlet_layout, int entrain, int steps,
                      int offset, double dt, double inlet, int seedbin,
                      double sw, const double *rh, const double *map,
                      const int *dest, const double *weight, const double *mass,
                      const double *speed, const double *diam,
                      const double *pars, double *conc, double *gas,
                      double *budgets, double *average) {
  if (n > MAX_BINS || layers * patches > MAX_ZONES)
    return 1;
  advance<<<cases, 32 * layers * patches>>>(
      n, h, layers, patches, pulse, inlet_layout, entrain, steps, offset, dt,
      inlet, seedbin, sw, rh, map, dest, weight, mass, speed, diam, pars, conc,
      gas, budgets, average);
  return cudaGetLastError();
}
