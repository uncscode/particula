// Continuous-time rejection sampling of distinct physical droplet pairs in one cell.
// Masses are kg, V is cm3, lookup kernels are cm3/s.
#include <algorithm>
#include <cmath>
#include <random>
extern "C" int evolve(double* mass, int count, double volume, double dt,
    double other, double middle, const double* kernel, const double* bounds,
    int bins, double mid_bound, unsigned long long seed) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> uniform(0.,1.);
    auto diameter=[](double m){return 1.e6*std::cbrt(6*m/(3.14159265358979323846*1000.));};
    auto index=[&](double m){return static_cast<int>(diameter(m));};
    int maximum=0;
    for(int i=0;i<count;i++)maximum=std::max(maximum,index(mass[i]));
    double time=0;
    while(count>=2){
        if(maximum>=bins)return -1;
        const double ceiling=std::max(other*bounds[maximum],middle*mid_bound);
        const double rate=ceiling*(double(count)*(count-1)/2)/volume;
        time-=std::log1p(-uniform(gen))/rate;
        if(time>=dt)break;
        int a=std::uniform_int_distribution<int>(0,count-1)(gen);
        int b=std::uniform_int_distribution<int>(0,count-2)(gen);
        if(b>=a)++b;
        double da=diameter(mass[a]),db=diameter(mass[b]);
        bool mid=da>=20 && da<=48 && db>=20 && db<=48;
        double k=kernel[index(mass[a])*bins+index(mass[b])]*(mid?middle:other);
        if(k>ceiling*(1+1.e-12))return -2;
        if(uniform(gen)*ceiling>=k)continue;
        mass[a]+=mass[b];maximum=std::max(maximum,index(mass[a]));
        mass[b]=mass[count-1];--count;
    }
    return count;
}
