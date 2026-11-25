// One CUDA block is one parcel. Each active lane follows one weighted particle.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include "organic.cuh"

constexpr double PI = 3.14159265358979323846, VOL = 4.0 * PI / 3;
constexpr double RD = 287.05799595894405, RV = 461.52280831345604, CP = 1005.;
constexpr double LV = 2.5e6, MW = .150, RHO_O = 1300., MW_W = .01801528;
constexpr int NP = 16, NO = 12;
__device__ double cube(double r) { return r * r * r; }
__device__ double psat(double t) {
    return 611.2 * exp(LV / RV * (1. / 273.15 - 1. / t));
}
__device__ double ratio(double p, double t, double q) {
    return p * q / ((RD / RV + q) * psat(t));
}
__device__ double volume(double p, double t, double q) { return (RD + q * RV) * t / p; }

// Reduce a pair of quantities; every lane, including padding, participates.
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

struct Particle {
    double a, a3, mass, nabs0;
    const double2 *table;
    double ix;
    int nodes;
    // The uncoated solution at the same water volume sets tension and baseline
    // activity. Added solute contributes partial volume and dilute Raoult effect.
    __device__ double eq(double u, double T, double nu, double *sigma = nullptr) const {
        double r = sqrt(u);
        double w = fmax(0., r * u - a3 - mass / (RHO_O * VOL));
        double x = log1p(w / a3) * ix / 3.;
        int j = min(nodes - 2, max(0, (int)x));
        double f = fmin(1., fmax(0., x - j));
        double2 l = table[j], h = table[j + 1];
        double aw = l.x + f * (h.x - l.x), sig = l.y + f * (h.y - l.y);
        if (sigma)
            *sigma = sig;
        aw = org_water_activity(aw, 1000. * VOL * w / MW_W, mass / MW, nu);
        return aw * exp(2. * sig / (RV * 1000. * T * r));
    }
    __device__ double critical(double T, double nu) const {
        // Global maximum scan, then refine the best interval in log water-size.
        double val = -1, bx = 0, stride = 8 / ix;
        for (int j = 0; j < nodes; j += 8) {
            double x = j / ix, r = cbrt(a3 * exp(3 * x) + mass / (RHO_O * VOL));
            double s = eq(r * r, T, nu);
            if (s > val) {
                val = s;
                bx = x;
            }
        }
        double l = fmax(0., bx - stride), h = fmin((nodes - 1) / ix, bx + stride);
        for (int j = 0; j < 24; j++) {
            double x1 = (2 * l + h) / 3., x2 = (l + 2 * h) / 3.;
            double r1 = cbrt(a3 * exp(3 * x1) + mass / (RHO_O * VOL));
            double r2 = cbrt(a3 * exp(3 * x2) + mass / (RHO_O * VOL));
            if (eq(r1 * r1, T, nu) < eq(r2 * r2, T, nu))
                l = x1;
            else
                h = x2;
        }
        return cbrt(a3 * exp(1.5 * (l + h)) + mass / (RHO_O * VOL));
    }
};

// Backward water-growth step at trial saturation. The derivative is also used
// in the scalar Newton solve for the heat/water feedback across all particles.
__device__ double water_step(const Particle &c, double u0, double u, double kgdt,
                             double rh, double t, double nu, double &slope) {
    double umin = pow(c.a3 + c.mass / (VOL * RHO_O), 2. / 3.);
    double hi = u0 + kgdt * rh;
    for (int it = 0; it < 12; it++) {
        double s = c.eq(u, t, nu);
        double du = u * 2.e-5;
        double ds = (c.eq(u + du, t, nu) - s) / du;
        double resid = u - u0 - kgdt * (rh - s);
        slope = kgdt / fmax(0.35, 1. + kgdt * ds);
        if (fabs(resid) < 1.e-10 * u + 2.e-23)
            break;
        if (resid > 0)
            hi = u;
        else
            umin = u;
        double trial = u - resid * slope / kgdt;
        u = (trial > umin && trial < hi) ? trial : 0.5 * (umin + hi);
    }
    return u;
}

__global__ void __launch_bounds__(1024, 1)
    run(const double *pars, const double *seed, const double2 *tabs,
        const double *velocities, double *diagnostics, double *individuals,
        double *histories, int n, int nodes, int nstep, int nh) {
    int i = threadIdx.x, box = blockIdx.x;
    bool present = i < n;
    const double *par = pars + box * NP;
    int idx = box * n + (present ? i : 0);
    // a, weight #/kg (unmixed endmember), initial local wet r, entrant flag,
    // background absorbing organic/acid moles per particle.
    const double *inp = seed + idx * 6;
    Particle c{inp[0],
               cube(inp[0]),
               inp[5],
               inp[4],
               tabs + (long long)idx * nodes,
               (nodes - 1) / log(1500.),
               nodes};
    bool incoming = inp[3] > 0.5;
    double weight = (present && !incoming) ? inp[1] : 0.;
    double rw = inp[2], u = rw * rw;
    double p = 90000., t = 285., q = (RD / RV) * psat(t) / (p - psat(t));
    double dt = par[0], entr = par[1], extRH = par[2],
           gt = par[3] * 1e-9 * volume(p, t, q);
    double nu = par[6], maximum = 0., maxerr = 0., omax = 0., minmargin = 1.e30, z = 0.;
    double2 initial =
        sum2(weight * VOL * 1000. * (cube(rw) - c.a3 - c.mass / (VOL * RHO_O)),
             weight * c.mass);
    double ql = initial.x;
    double qt = q + ql, gas = gt - initial.y, event_z = -1, maxenergy = 0,
           maximplicit = 0;
    bool mixed = false;
    int hc = 0;
    for (int step = 0; step < nstep; step++) {
        double dz = velocities[box * nstep + step] * dt;
        if (!mixed && entr > 0 && z >= 50. - 1.e-9) {
            double te = t + par[7], pe = extRH * psat(te);
            double qe = RD / RV * pe / (p - pe);
            double venv = volume(p, te, qe);
            // For incoming particles equilibrate to this environment before exchange.
            if (incoming) {
                double lo = c.a, hi = inp[2] * 2;
                for (int j = 0; j < 30; j++) {
                    double mid = (lo + hi) * .5;
                    if (c.eq(mid * mid, te, nu) < extRH)
                        lo = mid;
                    else
                        hi = mid;
                }
                rw = (lo + hi) * .5;
                u = rw * rw;
            }
            double ne =
                (present && incoming) ? inp[1] * venv : 0.; // entrant input #/m³
            double el = sum2(ne * VOL * 1000. * (cube(rw) - c.a3), 0).x;
            qt = (1. - entr) * qt + entr * (qe + el);
            gt = (1. - entr) * gt + entr * par[3] * par[8] * 1e-9 * venv;
            gas = (1. - entr) * gas + entr * par[3] * par[8] * 1e-9 * venv;
            weight = (1. - entr) * weight + entr * ne;
            q = (1. - entr) * q + entr * qe;
            t = (1. - entr) * t + entr * te;
            ql = qt - q;
            mixed = true;
            event_z = z;
        }
        bool live = present && (!incoming || mixed);
        double vol = volume(p, t, q);
        double dp = -(1. + qt) * 9.80665 * dz / vol;
        double td = t + vol * dp / CP;
        p += dp;
        z += dz;
        if (gt > 0.) {
            double sig, old = c.mass;
            c.eq(u, t, nu, &sig);
            double nwater =
                fmax(0., cube(rw) - c.a3 - old / (VOL * RHO_O)) * VOL * 1000. / MW_W;
            double abs = org_absorber_moles(nwater, c.nabs0, par[4]);
            double k = org_conductance(rw, t, p, 6e-6, par[10], MW);
            double ksat = org_saturation(t, par[5] * 1e-9, 298.15, 65000., MW) *
                          org_kelvin(rw, t, sig, MW, RHO_O);
            double cg = gas / vol, mass = old;
            for (int j = 0; j < 5; j++) {
                mass = org_target_mass(old, cg, dt, k, ksat, abs, MW);
                double b = abs * MW;
                double deriv =
                    dt * k / (1. + dt * k * ksat * b / ((b + mass) * (b + mass)));
                double2 m = sum2(weight * mass, weight * deriv);
                cg = fmin(gt / vol, fmax(0., cg - (vol * cg + m.x - gt) / (vol + m.y)));
            }
            mass = org_target_mass(old, cg, dt, k, ksat, abs, MW);
            double mo = sum2(weight * mass, 0).x;
            omax = fmax(omax, fabs(gt - vol * cg - mo));
            gas = cg * vol;
            if (live) {
                rw = cbrt(cube(rw) + (mass - old) / (VOL * RHO_O));
                u = rw * rw;
                c.mass = mass;
            }
        }
        double G =
            2. /
            (1000. *
             (RV * td / (2.11e-5 * pow(td / 273.15, 1.94) * (101325. / p) * psat(td)) +
              LV / (.001 * (4.39 + .071 * td) * td) * (LV / (RV * td) - 1.)));
        double u0 = u, x = u, rh = ratio(p, td, q), dl = 0;
        for (int it = 0; it < 5; it++) {
            double deriv = 0.;
            if (live)
                x = water_step(c, u0, x, G * dt, rh, td + LV / CP * dl, nu, deriv);
            double r = sqrt(x);
            double2 sum = sum2(weight * VOL * 1000. * (r * x - rw * u0),
                               weight * VOL * 1500. * r * deriv);
            dl = sum.x;
            double qnew = q - dl, tnew = td + LV / CP * dl;
            double sr = ratio(p, tnew, qnew);
            double slope = sr * ((RD / RV) / (qnew * (RD / RV + qnew)) +
                                 LV * LV / (CP * RV * tnew * tnew));
            double err = rh - sr;
            if (it == 4 || fabs(err) < 2e-11)
                maximplicit = fmax(maximplicit, fabs(err));
            rh -= err / (1. + slope * sum.y);
            if (fabs(err) < 2e-11)
                break;
        }
        rw = sqrt(x);
        u = x;
        q -= dl;
        t = td + LV / CP * dl;
        maxenergy = fmax(maxenergy, fabs((t - td) * CP - dl * LV));
        double wl =
            sum2(weight * VOL * 1000. * (rw * u - c.a3 - c.mass / (VOL * RHO_O)), 0).x;
        maxerr = fmax(maxerr, fabs(q + wl - qt));
        maximum = fmax(maximum, ratio(p, t, q) - 1);
        // Incoming particles have zero weight until their mixing event.
        if (step >= nstep - 1 - int(round(5. / dt)) &&
            ((nstep - 1 - step) % max(1, int(round(.5 / dt))) == 0 ||
             step == nstep - 1)) {
            double rc = c.critical(t, nu);
            minmargin = fmin(minmargin, log(rw / rc));
        }
        if ((step + 1) % (nstep / nh) == 0 && hc < nh) {
            double *h = histories + (box * nh + hc++) * 6;
            if (i == 0) {
                h[0] = z;
                h[1] = ratio(p, t, q) - 1;
                h[2] = t;
                h[3] = wl;
                h[4] = gas;
                h[5] = p;
            }
        }
    }
    if (present) {
        double *row = individuals + idx * 5;
        row[0] = weight / volume(p, t, q) * 1e-6;
        row[1] = minmargin;
        row[2] = rw;
        row[3] = c.mass;
        row[4] = weight;
    }
    if (i == 0) {
        double *out = diagnostics + box * NO;
        out[0] = maximum * 100;
        out[1] = maxerr;
        out[2] = omax;
        out[3] = gas;
        out[4] = gt;
        out[5] = event_z;
        out[6] = z;
        out[7] = p;
        out[8] = t;
        out[9] = q;
        out[10] = maxenergy;
        out[11] = maximplicit;
    }
}

// Hosts pass contiguous arrays; any failed allocation or launch ends the batch.
extern "C" int cloud_run(const double *params, const double *seed, const double *table,
                         const double *wind, double *out, double *particles,
                         double *hist, int boxes, int n, int nodes, int steps, int nh) {
    void *ptr[7] = {};
    const void *src[4] = {params, seed, table, wind};
    size_t b = boxes,
           sizes[7] = {b * NP * 8, b * n * 6 * 8, b * n * nodes * 16, b * steps * 8,
                       b * NO * 8, b * n * 5 * 8, b * nh * 6 * 8};
    cudaError_t e = cudaSuccess;
    for (int j = 0; j < 7 && !e; j++) {
        e = cudaMalloc(&ptr[j], sizes[j]);
        if (!e && j < 4)
            e = cudaMemcpy(ptr[j], src[j], sizes[j], cudaMemcpyHostToDevice);
    }
    if (!e) {
        run<<<boxes, 32 * ((n + 31) / 32)>>>(
            (double *)ptr[0], (double *)ptr[1], (double2 *)ptr[2], (double *)ptr[3],
            (double *)ptr[4], (double *)ptr[5], (double *)ptr[6], n, nodes, steps, nh);
        e = cudaGetLastError();
    }
    if (!e)
        e = cudaDeviceSynchronize();
    double *dst[3] = {out, particles, hist};
    for (int j = 4; j < 7 && !e; j++)
        e = cudaMemcpy(dst[j - 4], ptr[j], sizes[j], cudaMemcpyDeviceToHost);
    for (void *p : ptr)
        if (p)
            cudaFree(p);
    if (e)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
    return (int)e;
}
