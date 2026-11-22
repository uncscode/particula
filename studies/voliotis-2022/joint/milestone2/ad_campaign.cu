// Forward-mode derivatives of the fixed-topology Manchester bulk replay.
// Each thread carries one directional derivative for one chamber experiment.
// The Python driver checks this primal and its Jacobian against
// discovery/gpu.py.
#include <cmath>
#include <cuda_runtime.h>

struct Dual {
  double value, tangent;
  __device__ Dual(double value = 0., double tangent = 0.)
      : value(value), tangent(tangent) {}
};
__device__ Dual operator+(Dual a, Dual b) {
  return {a.value + b.value, a.tangent + b.tangent};
}
__device__ Dual operator-(Dual a, Dual b) {
  return {a.value - b.value, a.tangent - b.tangent};
}
__device__ Dual operator*(Dual a, Dual b) {
  return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}
__device__ Dual operator/(Dual a, Dual b) {
  const double ratio = a.value / b.value;
  return {ratio, (a.tangent - ratio * b.tangent) / b.value};
}
__device__ Dual exponential(Dual a) {
  const double value = exp(a.value);
  return {value, a.tangent * value};
}
__device__ Dual power(Dual a, double exponent) {
  const double value = pow(a.value, exponent);
  return {value, exponent * a.tangent * value / a.value};
}
__device__ Dual root(Dual a) {
  const double value = sqrt(a.value);
  return {value, a.tangent / (2. * value)};
}
__device__ Dual tanh_dual(Dual a) {
  const double value = tanh(a.value);
  return {value, a.tangent * (1. - value * value)};
}
__device__ Dual positive_part(Dual a) { return a.value > 0. ? a : Dual(0.); }

constexpr double pi = 3.141592653589793, temperature = double(298.15f);
constexpr double kb = 1.380649e-23, gravity = 9.80665;
constexpr double iso_ug_per_ppb =
    68.117 * 101325. / (8.314462618 * 298.15) * 1.e-3;

struct Transport {
  Dual area_ratio, wall, brownian;
};

// Scalar forms of rectangle_wall_loss_coefficient_from_transport_wp and
// brownian_equal_pair for the measured positive-radius chamber states.
__device__ Transport transport(Dual seed, Dual aerosol, Dual count,
                               const double *initial, Dual eddy) {
  const Dual mass = seed + aerosol;
  const Dual volume = seed / 1770. + aerosol / 1400.;
  // The archived Warp kernel casts these three expressions from fp32 literals.
  const Dual radius =
      power(double(3.e-9f) * volume / (double(4.f * float(pi)) * count),
            double(1.f / 3.f));
  const Dual kn = initial[6] / radius;
  const Dual slip = 1. + kn * (1.257 + 0.4 * exponential(-1.1 / kn));
  const Dual diffusivity =
      kb * temperature * slip / (6. * pi * initial[5] * radius);
  const Dual settling = 2. * radius * radius * (mass / volume) * slip *
                        gravity / (9. * initial[5]);
  const Dual scale = root(eddy * diffusivity);
  const Dual x = pi * settling / (4. * scale);
  const Dual xx = x * x;
  const Dual xcoth =
      x.value <= 1.e-3 ? 1. + xx / 3. - xx * xx / 45. + 2. * xx * xx * xx / 945.
                       : x / tanh_dual(x);
  const Dual wall = 4. * scale / pi * (1. / 3. + 1. / 2. + xcoth / 3.);
  const Dual speed = root(8. * kb * temperature /
                          (double(float(pi)) * mass * double(1.e-9f) / count));
  const Dual path = 8. * diffusivity / (pi * speed);
  const Dual twice_r = 2. * radius;
  const Dual g = (power(twice_r + path, 3.) -
                  power(twice_r * twice_r + path * path, 1.5)) /
                     (6. * radius * path) -
                 twice_r;
  const Dual kernel = 4. * pi * 2. * diffusivity * twice_r /
                      (twice_r / (twice_r + root(2. * g * g)) +
                       8. * diffusivity / (twice_r * sqrt(2.) * speed));
  return {4. * pi * radius * radius * count / double(0.001f), wall, kernel};
}

__device__ bool labelled_solid(int index) { return index >= 11 && index < 26; }
__device__ Dual suspended(const Dual *x) { return x[6] + x[7] + x[8] + x[9]; }
__device__ void transfer(Dual *x, const int *row, Dual k, double dt_h,
                         const double *mw) {
  const int a = row[0], b = row[1], destination = row[2], split = row[4];
  Dual moved;
  if (b < 0) {
    moved = x[a] * (1. - exponential((0. - k) * dt_h));
    x[a] = x[a] - moved;
  } else {
    const Dual na = x[a] / mw[a % 3];
    const Dual nb = x[b] / mw[b % 3];
    if (a == b) {
      const Dual z = 2. * k * dt_h * na;
      moved = x[a] * z / (1. + z);
      x[a] = x[a] - moved;
    } else {
      const Dual extent = k * dt_h * na * nb /
                          (1. + k * dt_h * (na.value > nb.value ? na : nb));
      const Dual left = extent * mw[a % 3], right = extent * mw[b % 3];
      x[a] = x[a] - left;
      x[b] = x[b] - right;
      moved = left + right;
    }
  }
  x[destination] = x[destination] + moved;
  if (labelled_solid(a))
    x[9] = x[9] - moved;
  if (labelled_solid(destination))
    x[9] = x[9] + moved;
  if (split >= 0) {
    const Dual secondary = 0.5 * moved;
    x[destination] = x[destination] - secondary;
    x[split] = x[split] + secondary;
    if (labelled_solid(destination))
      x[9] = x[9] - secondary;
    if (labelled_solid(split))
      x[9] = x[9] + secondary;
  }
}

// A direction id of -1 seeds each run's own dose; nonnegative ids are
// shared rates; -2-j seeds shared precursor/forcing parameter j. The Jacobian
// of other runs with respect to this dose is zero.
__global__ void campaign(const double *source, const double *observations,
                         const double *initial, const int *channels,
                         const int *active, const double *p,
                         const double *rates, const double *doses,
                         const int *directions, int runs, int ndir, int nactive,
                         int sub, double *output, double *minute_tangents) {
  const int direction = blockIdx.x * blockDim.x + threadIdx.x;
  const int run = blockIdx.y;
  if (direction >= ndir || run >= runs)
    return;
  const int wrt = directions[direction];
  const double *initial_run = initial + run * 7;
  const Dual dose(doses[run], wrt == -1);
  const double mw[3] = {184., 140., 118.};
  const double step = 60. / sub, dh = step / 3600.;
  const double vapor_survival = exp(-dh);
  Dual par[13];
  for (int j = 0; j < 13; ++j)
    par[j] = Dual(p[j], wrt == -2 - j);
  Dual mature[3], fresh[3];
  for (int s = 0; s < 3; ++s) {
    const Dual z = dh / par[s];
    mature[s] = 1. - exponential(0. - z);
    fresh[s] = 1. - mature[s] / z;
  }
  Dual x[31] = {}, seed(initial_run[0]), number(initial_run[1]);
  Dual mass_sse, number_sse;
  for (int minute = 0; minute < 300; ++minute) {
    const double *src = source + (run * 300 + minute) * 6;
    const double *obs = observations + (run * 300 + minute) * 4;
    const Dual old_mass = suspended(x), old_number = number / initial_run[1];
    const Dual iso = (src[2] + src[5]) * dose / iso_ug_per_ppb;
    Dual supply[3], extra[3];
    for (int s = 0; s < 3; ++s) {
      const Dual reacted = (src[s] + src[s + 3]) * dose / sub;
      const Dual gross =
          (par[s + 3] * src[s] + par[s + 6] * src[s + 3]) * dose / sub;
      const Dual retain = s < 2 ? 1. / (1. + par[12] * iso) : Dual(1.);
      supply[s] = gross * retain;
      extra[s] = gross * (1. - retain) + 0.1 * positive_part(reacted - gross);
    }
    for (int j = 0; j < sub; ++j) {
      const Transport t =
          transport(seed, suspended(x), number, initial_run, par[9]);
      const Dual keep = exponential(-step * t.wall);
      for (int s = 0; s < 3; ++s) {
        const Dual move = x[s] * mature[s] + supply[s] * fresh[s];
        x[s] = x[s] + supply[s] - move;
        x[s + 6] = x[s + 6] * keep + move * root(keep);
        x[s + 3] = x[s + 3] * vapor_survival + extra[s] * sqrt(vapor_survival);
      }
      x[9] = x[9] * keep;
      for (int i = 11; i < 26; ++i)
        x[i] = x[i] * keep;
      for (int i = 26; i < 31; ++i)
        x[i] = x[i] * vapor_survival;
      for (int i = 0; i < nactive; ++i) {
        const int h = active[i];
        // Keep the one-sided derivative at a zero rate. A zero primal
        // transfer gives the parent trajectory exactly.
        if (rates[h] == 0. && h != wrt)
          continue;
        const Dual k(rates[h], wrt == h);
        const int *reaction = channels + 5 * h;
        transfer(x, reaction, reaction[3] ? k * t.area_ratio : k, dh, mw);
      }
      number = number /
               (1. + 0.5 * t.brownian * number * (1. - keep) / t.wall) * keep;
      seed = seed * keep;
    }
    if (minute_tangents) {
      double *row =
          minute_tangents + ((run * ndir + direction) * 300 + minute) * 2;
      row[0] = suspended(x).tangent;
      row[1] = number.tangent / initial_run[1];
    }
    if (obs[0] >= 0.) {
      const Dual residual =
          (old_mass + (suspended(x) - old_mass) * obs[0] - obs[1]) /
          initial_run[2];
      mass_sse = mass_sse + residual * residual;
    }
    if (obs[2] >= 0.) {
      const Dual residual =
          old_number + (number / initial_run[1] - old_number) * obs[2] - obs[3];
      number_sse = number_sse + residual * residual;
    }
  }
  // A zero residual norm has a valid zero subgradient. Isoprene-only
  // runs can have both observed SOA and the fitted yield identically zero.
  const Dual m =
      mass_sse.value == 0. ? Dual(0.) : root(mass_sse / initial_run[3]);
  const Dual n =
      number_sse.value == 0. ? Dual(0.) : root(number_sse / initial_run[4]);
  double *result = output + (run * ndir + direction) * 4;
  result[0] = m.value;
  result[1] = n.value;
  result[2] = m.tangent;
  result[3] = n.tangent;
}

extern "C" int campaign_jvp(const double *source, const double *obs,
                            const double *initial, const int *channels,
                            const int *active, const double *p, const double *k,
                            const double *dose, const int *directions, int runs,
                            int ndir, int nactive, int sub, double *output,
                            double *minute_tangents, cudaStream_t stream) {
  campaign<<<dim3((ndir + 31) / 32, runs), 32, 0, stream>>>(
      source, obs, initial, channels, active, p, k, dose, directions, runs,
      ndir, nactive, sub, output, minute_tangents);
  return static_cast<int>(cudaGetLastError());
}
