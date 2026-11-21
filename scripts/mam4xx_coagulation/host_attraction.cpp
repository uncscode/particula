// Apply independently integrated number and donor-volume multipliers to
// the archived fine-pair coefficients. Number is per cm3, volume µm3/cm3.
#include "regime_core.inc"

extern "C" void advance_fine(double dt, const double *factor,
                              double *number, double *volume) {
    using namespace mam4::coagulation;
    constexpr double temperature=298.15, pressure=101325., density=1770.;
    constexpr double air=pressure/(8314.462618*temperature);
    double q[4]={}, old_q[4]={}, avg_q[4]={}, old_m[7][4]={}, mass[7][4]={};
    double aged[7][1]={}, diameter[2]={};
    for (int i=0; i<2; ++i) {
        int mode=1-i; // study order Aitken, accumulation
        double log_width=std::log(mam4::modes(mode).mean_std_dev);
        q[mode]=old_q[mode]=number[i]*1e6/air;
        old_m[0][mode]=mass[0][mode]=volume[i]*density/(1e12*air);
        diameter[i]=1e-6*std::cbrt(6*volume[i]/(std::acos(-1.)*number[i]
                                     *std::exp(4.5*log_width*log_width)));
    }
    double ij[3]={}, donor[3]={}, ii[3]={}, jj[3]={};
    getcoags_wrapper_f(temperature,pressure,diameter[0],diameter[1],1.6,1.8,
                       std::log(1.6),std::log(1.8),density,density,
                       ij[0],donor[0],ii[0],jj[0]);
    ii[0]*=air*factor[0];
    jj[0]*=air*factor[1];
    ij[0]*=air*factor[2];
    donor[0]*=air*factor[3];
    mam_coag_num_update(ij,ii,jj,dt,old_q,q,avg_q);
    mam_coag_aer_update(donor,dt,avg_q,old_m,mass,aged);
    for (int i=0; i<2; ++i) {
        number[i]=q[1-i]*air/1e6;
        volume[i]=mass[0][1-i]*air/density*1e12;
    }
}
