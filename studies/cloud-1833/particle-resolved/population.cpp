// One well-mixed independent box per host worker. Every particle has one
// sulfate inventory. Brownian mergers select actual pairs by Poisson thinning.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <cstdlib>
struct Particle{double mass;int bin;};
extern "C" int population(int boxes,int nt,int grid,int ni,int discrete,double volume,double step,double ws,
    const double* t,const double* acid,const double* mass,const double* diam,const double* cap,
    const double* wall,const double* kernel,const double* initial,const int* offset,
    const double* arrivals,const double* solute,const double* edges,int nb,unsigned seed,double* out,double* audit){
    double nmax=*std::max_element(kernel,kernel+grid*grid);
    #pragma omp parallel for schedule(dynamic)
    for(int box=0;box<boxes;box++){
        std::mt19937_64 rng(seed+box*104729u);
        std::uniform_real_distribution<double> uniform(0.,1.);
        auto poisson=[&](double x){return x>0?std::poisson_distribution<int>(x)(rng):0;};
        auto locate=[&](double m){return std::max(0,int(std::upper_bound(mass,mass+grid,m)-mass)-1);};
        std::vector<Particle> pop;
        double available=0,removed=0,mergers=0,number_in=0,number_lost=0,exposure=0;
        auto add=[&](double m){pop.push_back({m,locate(m)});available+=m;number_in++;};
        for(int k=0;k<ni;k++){
            int count=poisson(initial[k]*volume);
            for(int a=0;a<count;a++)add(initial[ni+k]);
        }
        int it=offset[box];
        for(int n=0;n<nt;n++){
            for(auto &v:pop){
                int k=v.bin;
                double frac=(v.mass-mass[k])/(mass[k+1]-mass[k]);
                double size=diam[k]+frac*(diam[k+1]-diam[k]);
                int b=int(std::upper_bound(edges,edges+nb+1,size)-edges)-1;
                if(b>=0 && b<nb)out[(box*nt+n)*(nb+1)+b]++;
                if(size>=9.7 && size<=edges[nb])out[(box*nt+n)*(nb+1)+nb]++;
            }
            if(n==nt-1)break;
            int steps=(int)std::ceil((t[n+1]-t[n])/step);
            double h=(t[n+1]-t[n])/steps;
            for(int s=0;s<steps;s++){
                double lo=t[n]+s*h,hi=lo+h;
                double gas=acid[n]+(acid[n+1]-acid[n])*(s+.5)/steps;
                int prior=pop.size();
                int start=it;
                while(it<offset[box+1] && arrivals[it]<=hi){add(solute[it++]);}
                size_t dst=0;
                for(size_t j=0;j<pop.size();j++){
                    auto v=pop[j];int k=v.bin;
                    double dt=j<(size_t)prior?h:std::max(0.,hi-arrivals[start+j-prior]);
                    double f=(v.mass-mass[k])/(mass[k+1]-mass[k]);
                    double hazard=ws*(wall[k]+f*(wall[k+1]-wall[k]))+.0001754037;
                    if(uniform(rng)<-std::expm1(-dt*hazard)){removed+=v.mass;number_lost++;continue;}
                    double expected=dt*gas*(cap[k]+f*(cap[k+1]-cap[k]));
                    double gain=discrete?poisson(expected):expected;
                    v.mass+=gain;available+=gain;
                    if(v.mass>=mass[grid-1]){removed+=v.mass;number_lost++;continue;}
                    v.bin=locate(v.mass);pop[dst++]=v;
                }
                pop.resize(dst);
                // Poisson proposal clock at a global upper bound. Decreasing
                // pair count after accepted mergers is included in thinning.
                int count=pop.size();
                int trials=poisson(nmax*count*(count-1)*.5*h/volume);
                while(trials-- && pop.size()>1){
                    int m=pop.size();
                    int a=int(uniform(rng)*m),b=int(uniform(rng)*(m-1));if(b>=a)b++;
                    double prob=kernel[pop[a].bin*grid+pop[b].bin]/nmax*(double(m)*(m-1)/(double(count)*(count-1)));
                    if(uniform(rng)>=prob)continue;
                    if(a>b)std::swap(a,b);
                    double merged=pop[a].mass+pop[b].mass;pop[b]=pop.back();pop.pop_back();mergers++;
                    if(merged>=mass[grid-1]){removed+=merged;number_lost++;pop[a]=pop.back();pop.pop_back();}
                    else pop[a]={merged,locate(merged)};
                }
            }
        }
        double left=0;for(auto &v:pop)left+=v.mass;
        audit[box*3]=std::abs(available-removed-left)/std::max(1.,available);
        audit[box*3+1]=number_in-number_lost-mergers-pop.size();audit[box*3+2]=mergers;
    }
    return 0;
}
