// One well-mixed independent box per host worker. Every particle has one
// sulfate inventory. Brownian mergers select actual pairs by Poisson thinning.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <cstdlib>
struct Particle{double mass;int bin;int q;};
extern "C" int population(int boxes,int nt,int grid,int ni,int discrete,double volume,double step,double ws,
    const double* t,const double* acid,const double* mass,const double* diam,const double* cap,
    const double* wall,const double* kernel,const double* initial,const int* offset,
    const double* arrivals,const double* solute,const double* edges,int nb,unsigned seed,double* out,double* audit,
    int charge,const double* rates,const double* ions,const double* signs,double* signed_out,double* charge_audit,double ais_upper){
    double nmax=*std::max_element(kernel,kernel+grid*grid);
    #pragma omp parallel for schedule(dynamic)
    for(int box=0;box<boxes;box++){
        std::mt19937_64 rng(seed+box*104729u);
        std::uniform_real_distribution<double> uniform(0.,1.);
        auto poisson=[&](double x){return x>0?std::poisson_distribution<int>(x)(rng):0;};
        auto locate=[&](double m){return std::max(0,int(std::upper_bound(mass,mass+grid,m)-mass)-1);};
        std::vector<Particle> pop;
        double available=0,removed=0,mergers=0,number_in=0,number_lost=0,exposure=0;
        int net_in=0,net_out=0,field_count=0,attached=0,neutralized=0;
        auto add=[&](double m,int q){net_in+=q;pop.push_back({m,locate(m),q});available+=m;number_in++;};
        for(int k=0;k<ni;k++){
            int count=poisson(initial[k]*volume);
            for(int a=0;a<count;a++)add(initial[ni+k],0);
        }
        int it=offset[box];
        for(int n=0;n<nt;n++){
            for(auto &v:pop){
                int k=v.bin;
                double frac=(v.mass-mass[k])/(mass[k+1]-mass[k]);
                double size=diam[k]+frac*(diam[k+1]-diam[k]);
                int b=int(std::upper_bound(edges,edges+nb+1,size)-edges)-1;
                if(b>=0 && b<nb)out[(box*nt+n)*(nb+1)+b]++;
                if(v.q && size>=2.2 && size<=ais_upper){
                    int band=(size>=2.9)+(size>=5.)+(size>=9.7);
                    signed_out[8*(box*nt+n)+2*band+(v.q>0)]++;
                }
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
                while(it<offset[box+1] && arrivals[it]<=hi){add(solute[it],int(signs[it]));++it;}
                size_t dst=0;
                for(size_t j=0;j<pop.size();j++){
                    auto v=pop[j];int k=v.bin;
                    double dt=j<(size_t)prior?h:std::max(0.,hi-arrivals[start+j-prior]);
                    double f=(v.mass-mass[k])/(mass[k+1]-mass[k]);
                    double hazard=ws*(wall[k]+f*(wall[k+1]-wall[k]))+.0001754037;
                    if(uniform(rng)<-std::expm1(-dt*hazard)){removed+=v.mass;net_out+=v.q;number_lost++;continue;}
                    if(charge){
                        int bin=n+int(s+.5>=steps*.5);
                        const double *air=&ions[3*bin],*c=&rates[4*k];
                        // A charge jump changes the next competing hazards immediately.
                        double elapsed=0;bool collected=false;
                        while(elapsed<dt){
                            double neg=(v.q==0?c[0]:v.q>0?c[1]:0.)*air[0];
                            double pos=(v.q==0?c[0]:v.q<0?c[2]:0.)*air[1];
                            double field=std::abs(v.q)*c[3]*air[2];
                            double total=neg+pos+field;
                            if(total<=0)break;
                            elapsed+=-std::log(1.-uniform(rng))/total;
                            if(elapsed>=dt)break;
                            double r=uniform(rng)*total;
                            if(r>=neg+pos){field_count++;collected=true;break;}
                            if(v.q==0)attached++;else neutralized++;
                            int jump=r<neg?-1:1;
                            net_in+=jump;v.q+=jump;
                        }
                        if(collected){
                            removed+=v.mass;net_out+=v.q;number_lost++;continue;
                        }
                    }
                    double expected=dt*gas*(cap[k]+f*(cap[k+1]-cap[k]));
                    double gain=discrete?poisson(expected):expected;
                    v.mass+=gain;available+=gain;
                    if(v.mass>=mass[grid-1]){removed+=v.mass;net_out+=v.q;number_lost++;continue;}
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
                    int merged_q=pop[a].q+pop[b].q;double merged=pop[a].mass+pop[b].mass;pop[b]=pop.back();pop.pop_back();mergers++;
                    if(merged>=mass[grid-1]){removed+=merged;net_out+=merged_q;number_lost++;pop[a]=pop.back();pop.pop_back();}
                    else pop[a]={merged,locate(merged),merged_q};
                }
            }
        }
        int net_final=0;
        for(auto &v:pop)net_final+=v.q;
        charge_audit[4*box]=net_in-net_out-net_final;charge_audit[4*box+1]=field_count;
        charge_audit[4*box+2]=attached;charge_audit[4*box+3]=neutralized;
        double left=0;for(auto &v:pop)left+=v.mass;
        audit[box*3]=std::abs(available-removed-left)/std::max(1.,available);
        audit[box*3+1]=number_in-number_lost-mergers-pop.size();audit[box*3+2]=mergers;
    }
    return 0;
}
