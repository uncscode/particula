// Solar scalar discrete ordinates. Thin-cell single scattering followed by
// layer doubling, then exact matrix adding for vertically distinct layers.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

template <int N> struct Mat {
  double a[N][N]{};
  static Mat identity() {
    Mat m;
    for (int i = 0; i < N; ++i)
      m.a[i][i] = 1.;
    return m;
  }
};
template <int N> using Vec = std::array<double, N>;
template <int N> inline Mat<N> operator*(const Mat<N> &a, const Mat<N> &b) {
  Mat<N> c;
  for (int i = 0; i < N; ++i)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        c.a[i][j] += a.a[i][k] * b.a[k][j];
  return c;
}
template <int N> inline Mat<N> operator+(const Mat<N> &a, const Mat<N> &b) {
  Mat<N> c;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      c.a[i][j] = a.a[i][j] + b.a[i][j];
  return c;
}
template <int N>
inline Vec<N> mul(const Mat<N> &a,
                  const std::array<double, static_cast<size_t>(N)> &b) {
  Vec<N> c{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      c[i] += a.a[i][j] * b[j];
  return c;
}
template <size_t N>
inline std::array<double, N> plus(const std::array<double, N> &a,
                                  const std::array<double, N> &b,
                                  double scale = 1.) {
  std::array<double, N> c;
  for (size_t i = 0; i < N; ++i)
    c[i] = a[i] + scale * b[i];
  return c;
}
template <int N> Mat<N> transport_inverse(Mat<N> p) {
  // Invert I-p with partial pivoting. p is the product of the facing
  // reflection operators. The open upper and lower boundaries allow escape.
  Mat<N> r = Mat<N>::identity();
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      p.a[i][j] = (i == j) - p.a[i][j];
  for (int i = 0; i < N; ++i) {
    int pivot = i;
    for (int j = i + 1; j < N; ++j)
      if (std::abs(p.a[j][i]) > std::abs(p.a[pivot][i]))
        pivot = j;
    if (pivot != i) {
      std::swap(p.a[pivot], p.a[i]);
      std::swap(r.a[pivot], r.a[i]);
    }
    auto f = p.a[i][i];
    for (int j = 0; j < N; ++j) {
      p.a[i][j] /= f;
      r.a[i][j] /= f;
    }
    for (int k = 0; k < N; ++k)
      if (k != i) {
        f = p.a[k][i];
        for (int j = 0; j < N; ++j) {
          p.a[k][j] -= f * p.a[i][j];
          r.a[k][j] -= f * r.a[i][j];
        }
      }
  }
  return r;
}
template <int N> struct Layer {
  Mat<N> rt, rb, td, tu;
  Vec<N> up{}, down{};
  double beam = 1.;
};
template <int N> void add_below(Layer<N> &a, const Layer<N> &b) {
  auto h = transport_inverse(a.rb * b.rt);
  auto htd = h * a.td;
  auto hrbtu = h * a.rb * b.tu;
  auto inter = mul(h, plus(a.down, mul(a.rb, b.up), a.beam));
  auto top = plus(a.up, mul(a.tu, plus(mul(b.rt, inter), b.up, a.beam)));
  auto down = plus(mul(b.td, inter), b.down, a.beam);
  auto rt = a.rt + a.tu * b.rt * htd;
  auto rb = b.rb + b.td * hrbtu;
  auto tu = a.tu * (b.tu + b.rt * hrbtu);
  a.td = b.td * htd;
  a.tu = tu;
  a.rt = rt;
  a.rb = rb;
  a.up = top;
  a.down = down;
  a.beam *= b.beam;
}
template <int N>
Layer<N> slab(const double *stat, int order, double mu0, const double *pos,
              const double *weight, const double *leg, const double *sun,
              double thin) {
  Layer<N> s;
  // Delta-M removes the order-L forward lobe; absorption (ext-sca) is
  // preserved.
  double forward = stat[1 + order], tau = stat[0] - forward;
  if (tau <= 1e-14) {
    s.td = s.tu = Mat<N>::identity();
    return s;
  }
  double c[16];
  for (int ell = 0; ell < order; ++ell)
    c[ell] = (2 * ell + 1) * (stat[1 + ell] - forward) / tau;
  int steps = std::max(
      0, int(std::ceil(std::log2(tau / (thin * std::min(pos[0], mu0))))));
  double dt = std::ldexp(tau, -steps);
  s.beam = std::exp(-dt / mu0);
  double beammean = -std::expm1(-dt / mu0) / (dt / mu0);
  for (int i = 0; i < N; ++i) {
    double emit = -std::expm1(-dt / pos[i]);
    double weighted[16];
    double sup = 0., sdown = 0.;
    for (int l = 0; l < order; ++l) {
      weighted[l] = c[l] * leg[i * order + l];
      double src = weighted[l] * sun[l];
      sdown += src;
      sup += (l % 2 ? -src : src);
    }
    s.up[i] = emit * beammean * sup / (2 * mu0);
    s.down[i] = emit * beammean * sdown / (2 * mu0);
    for (int j = 0; j < N; ++j) {
      double same = 0., opp = 0.;
      for (int l = 0; l < order; ++l) {
        double v = weighted[l] * leg[j * order + l];
        same += v;
        opp += (l % 2 ? -v : v);
      }
      // Half-cell incident attenuation: second-order thin-slab source.
      double incoming = -std::expm1(-dt / pos[j]) / (dt / pos[j]);
      s.rt.a[i][j] = emit * .5 * opp * weight[j] * incoming;
      s.td.a[i][j] =
          (i == j) * (1 - emit) + emit * .5 * same * weight[j] * incoming;
    }
  }
  for (int k = 0; k < steps; ++k) {
    auto h = transport_inverse(s.rt * s.rt);
    auto ht = h * s.td;
    auto inter = mul(h, plus(s.down, mul(s.rt, s.up), s.beam));
    auto top = plus(s.up, mul(s.td, plus(mul(s.rt, inter), s.up, s.beam)));
    auto down = plus(mul(s.td, inter), s.down, s.beam);
    auto rt = s.rt + s.td * s.rt * ht;
    s.td = s.td * ht;
    s.rt = rt;
    s.beam *= s.beam;
    s.up = top;
    s.down = down;
  }
  s.rb = s.rt;
  s.tu = s.td;
  return s;
}
template <int N>
void integrate(int jobs, int layers, int stride, int order,
               const double *moments, const double *mu0, const double *pos,
               const double *weight, const double *leg, const double *sun,
               double thin, double *outputs) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < jobs; ++i) {
    Layer<N> col;
    for (int j = 0; j < layers; ++j) {
      auto part = slab<N>(moments + (i * layers + j) * stride, order, mu0[i],
                          pos, weight, leg, sun + i * order, thin);
      if (j == 0)
        col = part;
      else
        add_below(col, part);
    }
    auto out = outputs + 6 * i;
    for (int channel = 0; channel < 6; ++channel)
      out[channel] = 0.;
    for (int j = 0; j < N; ++j) {
      double flux = weight[j] * pos[j];
      out[0] += flux * col.up[j];
      out[1] += flux * col.down[j];
      for (int k = 0; k < N; ++k) {
        out[2] += 2 * flux * col.rt.a[j][k];
        out[3] += 2 * flux * col.td.a[j][k];
        out[4] += 2 * flux * col.rb.a[j][k];
        out[5] += 2 * flux * col.tu.a[j][k];
      }
    }
    out[1] += col.beam;
  }
}
extern "C" int solve(int streams, int jobs, int layers, int stride, int order,
                     const double *moments, const double *mu0,
                     const double *pos, const double *weight, const double *leg,
                     const double *sun, double thin, double *outputs) {
  if (order > 16 || order < 2 || layers < 1 || order > streams * 2 ||
      stride < order + 2)
    return 1;
#define CASE(N)                                                                \
  case N:                                                                      \
    integrate<N>(jobs, layers, stride, order, moments, mu0, pos, weight, leg,  \
                 sun, thin, outputs);                                          \
    break;
  switch (streams) {
    CASE(2) CASE(4) CASE(6) CASE(8) CASE(16) default : return 2;
  }
  return 0;
}
