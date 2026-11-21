// Standalone host scaffolding for the archived scalar coagulation routines.
// The generator places their unchanged function bodies in modal_core.inc.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace haero {
using Real = double;
using std::exp;
using std::log;
using std::pow;
using std::round;
using std::sqrt;
inline Real min(Real a, Real b) { return std::min(a, b); }
inline Real max(Real a, Real b) { return std::max(a, b); }
inline Real cube(Real a) { return a * a * a; }
inline Real square(Real a) { return a * a; }
constexpr Real epsilon() { return std::numeric_limits<Real>::epsilon(); }
struct Constants {
    static constexpr Real pressure_stp = 101325.;
    static constexpr Real freezing_pt_h2o = 273.15;
};
} // namespace haero
namespace mam4 {
using Real = double;
enum class ModeIndex { Accumulation, Aitken, Coarse, PrimaryCarbon };
struct Mode { Real mean_std_dev; };
inline Mode modes(int i) {
    constexpr Real widths[] = {1.8, 1.6, 1.8, 1.60000002384186};
    return {widths[i]};
}
struct AeroConfig {
    static constexpr int num_modes() { return 4; }
    static constexpr int num_aerosol_ids() { return 7; }
    static constexpr int max_agepair() { return 1; }
};
struct Coagulation {
    static constexpr int max_coagpair = 3;
    static constexpr int i_agepair_pca = 0;
};
#define KOKKOS_INLINE_FUNCTION inline
#include "modal_core.inc"
} // namespace mam4

int main() {
    constexpr double pi = 3.141592653589793;
    constexpr double temp = 298.15, pressure = 101325., rho = 1770.;
    constexpr double air = pressure / (8314.462618 * temp); // kmol/m3
    std::cout << std::setprecision(17);
    std::cout << "acc_nm,ratio,dt_s,time_s,aitken_m3,accum_m3,aitken_mass_kg_m3,accum_mass_kg_m3,aitken_dg_m,accum_dg_m,loss_m3_s\n";
    for (double acc : {60.,110.,220.}) {
        for (double ratio : {.25,1.,4.}) {
            for (int dt : {10,30,60}) {
                double dg[] = {acc*1e-9,26e-9,2e-6,50e-9};
                double density[] = {rho,rho,rho,rho};
                double number[] = {2e11/(1+ratio)/air,2e11*ratio/(1+ratio)/air,0,0};
                double mass[7][4] = {}, transferred[7][1] = {};
                for (int i : {0,1}) {
                    auto width = std::log(mam4::modes(i).mean_std_dev);
                    mass[0][i] = rho*pi/6 * number[i] * std::pow(dg[i],3) * std::exp(4.5*width*width);
                }
                for (int time=0; time<=3600; time+=dt) {
                    for (int i : {0,1}) {
                        auto width = std::log(mam4::modes(i).mean_std_dev);
                        dg[i] = std::cbrt(6*mass[0][i]/(rho*pi*number[i]*std::exp(4.5*width*width)));
                    }
                    if (time%600==0) {
                        double inter, volume, intra_i, intra_j;
                        const double w_i = mam4::modes(1).mean_std_dev;
                        const double w_j = mam4::modes(0).mean_std_dev;
                        mam4::coagulation::getcoags_wrapper_f(temp,pressure,dg[1],dg[0],w_i,w_j,
                            std::log(w_i),std::log(w_j),rho,rho,inter,volume,intra_i,intra_j);
                        const double ni=number[1]*air, nj=number[0]*air;
                        const double loss=intra_i*ni*ni+intra_j*nj*nj+inter*ni*nj;
                        std::cout<<acc<<','<<ratio<<','<<dt<<','<<time<<','<<ni<<','<<nj<<','
                            <<mass[0][1]*air<<','<<mass[0][0]*air<<','<<dg[1]<<','<<dg[0]<<','<<loss<<'\n';
                    }
                    if (time<3600) {
                        mam4::coagulation::mam_coag_1subarea(dt,temp,pressure,air,dg,density,number,mass,transferred);
                    }
                }
            }
        }
    }
}
