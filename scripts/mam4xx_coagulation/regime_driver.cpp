// Archived host types, scalar coagulation routine and prescribed widths.
#include "regime_core.inc"
#include <array>

int main() {
    constexpr double temp=298.15, pressure=101325., rho=1770.;
    constexpr double air=pressure/(8314.462618*temp);
    const double pi=std::acos(-1.);
    // Diameter in micrometres, geometric standard deviation and number/cm3.
    using Pair=std::array<double,6>;
    std::array<Pair,3> regimes{{{.018,1.4,310,.075,1.6,70},
                                {.016,1.6,1000,.067,2.1,800},
                                {.014,1.8,106000,.054,2.16,32000}}};
    std::cout<<std::setprecision(17)
             <<"regime,dt_s,time_s,n_ait_cm3,n_acc_cm3,v_ait_um3_cm3,v_acc_um3_cm3\n";
    for (int regime=0;regime<3;++regime) {
        const auto &p=regimes[regime];
        for (int dt : {10,30}) {
            double dg[]={p[3]*1e-6,p[0]*1e-6,2e-6,50e-9};
            double density[]={rho,rho,rho,rho};
            double number[]={p[5]*1e6/air,p[2]*1e6/air,0,0};
            double mass[7][4]={},transferred[7][1]={};
            for (int i : {0,1}) {
                const double log_width=std::log(i==0 ? p[4] : p[1]);
                mass[0][i]=rho*pi/6*number[i]*std::pow(dg[i],3)
                           *std::exp(4.5*log_width*log_width);
            }
            for (int time=0; time<=86400; time+=dt) {
                for (int i : {0,1}) {
                    const double w=std::log(mam4::modes(i).mean_std_dev);
                    dg[i]=std::cbrt(6*mass[0][i]/(rho*pi*number[i]*std::exp(4.5*w*w)));
                }
                if (time==0 || time==3600 || time==21600 || time==86400)
                    std::cout<<regime<<','<<dt<<','<<time<<','<<number[1]*air/1e6<<','
                             <<number[0]*air/1e6<<','<<mass[0][1]*air/rho*1e12<<','
                             <<mass[0][0]*air/rho*1e12<<'\n';
                if (time<86400)
                    mam4::coagulation::mam_coag_1subarea(dt,temp,pressure,air,dg,density,
                                                         number,mass,transferred);
            }
        }
    }
}
