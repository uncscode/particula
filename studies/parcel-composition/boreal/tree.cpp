// Nested inventory-preserving median splits for equal-count particle slots.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

struct Cell {
  int lo, hi, axis, priority;
  double scatter;
};
extern "C" int build(const double *state, const double *feat, const int *flags,
                     const int *ends, int bins, int cap, int rule, int *order,
                     double *means, int *meta) {
  // meta columns: bin, first cap, first inactive cap, first slot, slot count.
  int nodes = 0;
  for (int b = 0; b < bins; ++b) {
    if (ends[b] == ends[b + 1])
      continue;
    std::vector<Cell> cells;
    const int origin = nodes;
    std::priority_queue<std::pair<std::pair<int, double>, int>> queue;
    auto add = [&](int lo, int hi, int born) {
      const int id = nodes++, n = hi - lo;
      double s[9] = {}, sq[9] = {};
      int bc = 0, young = 0;
      for (int pos = lo; pos < hi; ++pos) {
        const int i = order[pos];
        bc += (flags[i] & 1) != 0;
        young += (flags[i] & 2) != 0;
        for (int j = 0; j < 9; ++j) {
          means[9 * id + j] += state[9 * i + j];
          double x = feat[9 * i + j];
          s[j] += x;
          sq[j] += x * x;
        }
      }
      int axis = 0;
      double vmax = -1;
      for (int j = 0; j < 9; ++j) {
        means[9 * id + j] /= n;
        const double v = sq[j] - s[j] * s[j] / n;
        if (v > vmax) {
          vmax = v;
          axis = j;
        }
      }
      int priority =
          rule && bc && bc < n ? 2 : (rule == 2 && young && young < n ? 1 : 0);
      cells.push_back({lo, hi, axis, priority, vmax});
      int *out = meta + 5 * id;
      out[0] = b;
      out[1] = born;
      out[2] = cap + 1;
      out[3] = lo;
      out[4] = n;
      if (n > 1)
        queue.push({{priority, vmax}, id});
    };
    add(ends[b], ends[b + 1], 1);
    for (int k = 2; k <= cap && !queue.empty(); ++k) {
      int id = queue.top().second;
      queue.pop();
      const Cell c = cells[id - origin];
      int mid = (c.hi + c.lo) / 2;
      if (c.priority) {
        int flag = c.priority == 2 ? 1 : 2;
        mid = std::partition(order + c.lo, order + c.hi,
                             [&](int i) { return (flags[i] & flag) != 0; }) -
              order;
      } else {
        std::nth_element(
            order + c.lo, order + mid, order + c.hi, [&](int a, int z) {
              double x = feat[9 * a + c.axis], y = feat[9 * z + c.axis];
              return x == y ? a < z : x < y;
            });
      }
      meta[id * 5 + 2] = k;
      add(c.lo, mid, k);
      add(mid, c.hi, k);
    }
  }
  return nodes;
}

// Inputs sorted by representative critical supersaturation. left/right give the
// resolved counts at both sides of each jump; simultaneous jumps are combined.
extern "C" void scores(int len, int cap, int n, const int *meta,
                       const double *x, const int *left, const int *right,
                       const int *channel, const double *area,
                       const double *refarea, const int *refcurve,
                       const double low, const double high, double *output) {
  for (int k = 1; k <= cap; ++k) {
    int channels[32] = {}, prior = 0, rows = 0, maxerr = 0, atlow = 0,
        athigh = 0;
    double a[3] = {};
    for (int i = 0; i < len;) {
      const double value = x[i];
      int end = i + 1;
      while (end < len && x[end] == value)
        ++end;
      int group = 0;
      for (int j = i; j < end; ++j) {
        const int *m = meta + 5 * j;
        if (m[1] > k || m[2] <= k)
          continue;
        group += m[4];
        ++rows;
        for (int l = 0; l < 3; ++l)
          a[l] += area[3 * j + l] * m[4];
      }
      if (group) {
        if (value > low && value <= high)
          maxerr = std::max(maxerr, std::abs(prior - left[i]));
        prior += group;
        channels[channel[i]] += group;
        if (value >= low && value <= high)
          maxerr = std::max(maxerr, std::abs(prior - right[i]));
      }
      i = end;
    }
    double mse = 0;
    for (int s = 0, c = 0; s < 31; ++s) {
      c += channels[s];
      int e = c - refcurve[s];
      mse += double(e) * e;
      if (s == 0 || s == 30)
        maxerr = std::max(maxerr, std::abs(e));
    }
    double *o = output + (k - 1) * 6;
    o[0] = 100 * std::sqrt(mse / 31) / n;
    o[1] = 100. * maxerr / n;
    for (int j = 0; j < 3; ++j)
      o[j + 2] = refarea[j] > 0 ? 100 * std::abs(a[j] / refarea[j] - 1) : a[j];
    o[5] = rows;
  }
}
