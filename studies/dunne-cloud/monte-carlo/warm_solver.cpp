// Conservative finite-volume SSPRK2 replay of sectional.py, with draw-local coefficients.
// Physical vertices, wall hazards, kernel and measurement projection come from Python.
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" int replay(int bins, int times, double max_step, double dx,
        const double *seconds, const double *initial, const double *boundary,
        const double *weights, const double *velocity, const double *hazard,
        const double *kernel, const int *higher, const double *fraction,
        double coag, int events, const double *switch_time, const double *removed, double *out) {
    std::vector<double> n(initial, initial+bins), pred(bins), rhs(bins), vel(bins+1);
    const int stride = bins+1;
    auto display = [&](int t) {
        for (int band=0; band<3; ++band) {
            double sum = 0;
            for (int i=0; i<bins; ++i) sum += weights[band*bins+i]*n[i];
            out[3*t+band] = sum;
        }
    };
    auto evaluate = [&](const std::vector<double>& y, double feed) {
        double left_flux = vel[0]*feed;
        for (int i=0; i<bins; ++i) {
            double slope = 0.0;
            if (i>0 && i+1<bins) {
                const double l = y[i]-y[i-1], r = y[i+1]-y[i];
                if (l*r>0.0) slope = std::copysign(std::min(std::abs(l),std::abs(r)),l);
            }
            const double right_flux = vel[i+1]*(y[i]+0.5*slope)/dx;
            rhs[i] = left_flux - right_flux - hazard[i]*y[i];
            left_flux = right_flux;
        }
        int p=0;
        for (int i=0; i<bins; ++i) {
            double debit = 0, number = y[i]*coag;
            for (int j=i; j<bins; ++j, ++p) {
                const double collisions = kernel[p]*number*y[j];
                debit += collisions;
                rhs[j] -= collisions;
                const int h = higher[p];
                const double upper = collisions*fraction[p];
                if (h<bins) rhs[h] += upper;
                rhs[h-1] += collisions-upper;
            }
            rhs[i] -= debit;
        }
    };
    int event = 0;
    display(0);
    for (int t=0; t<times-1; ++t) {
        const double delta = seconds[t+1]-seconds[t];
        const double *v0 = velocity+t*stride, *v1=v0+stride;
        const double fastest = *std::max_element(v0, v1+stride);
        const int steps = std::max({1, (int)std::ceil(delta/max_step),
                                      (int)std::ceil(fastest*delta/(0.4*dx))});
        const double dt = delta/steps;
        for (int s=0; s<steps; ++s) {
            const double w = (s+0.5)/steps;
            for (int i=0; i<=bins; ++i) vel[i] = (1-w)*v0[i]+w*v1[i];
            const double feed = (1-w)*boundary[t]+w*boundary[t+1];
            const double end = seconds[t] + (s+1)*dt;
            while (event<events && switch_time[event]<=end) {
                for (int i=0; i<bins; ++i) n[i] *= 1-removed[event*bins+i];
                ++event;
            }
            evaluate(n,feed);
            for (int i=0; i<bins; ++i) pred[i] = n[i]+dt*rhs[i];
            evaluate(pred,feed);
            for (int i=0; i<bins; ++i) {
                n[i] = 0.5*(n[i]+pred[i]+dt*rhs[i]);
                if (!(n[i]>=-1e-8) || !std::isfinite(n[i])) return 1+t;
            }
        }
        display(1+t);
    }
    return 0;
}
