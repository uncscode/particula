// One physical cluster per CUDA thread. The prescribed acid varies linearly
// between archive samples. Wall and dilution loss compete against acid uptake.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
__device__ double uniform01(uint64_t &s){
    s^=s>>12; s^=s<<25; s^=s>>27;
    return ((s*2685821657736338717ULL >> 11)+0.5)*1.1102230246251565e-16;
}
__global__ void track(int n,int nt,int discrete,double du,double start,double end,
 const double* tk,const double* acid,const double* cap,const double* lose,
 const double* birth,const int* first,const uint64_t* seeds,double* result){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    uint64_t rng=seeds[i];
    double now=birth[i],u=start,death=-log(uniform01(rng));
    int j=first[i]; int status=0;
    while(u<end && j<nt-1){
        double inc=discrete?1.:fmin(.25,end-u);
        double pos=(u-start+(discrete?0.:.5*inc))/du;
        int a=(int)floor(pos);
        double f=pos-a;
        double rate=cap[a]+f*(cap[a+1]-cap[a]);
        double loss=lose[a]+f*(lose[a+1]-lose[a]);
        double exposure=(discrete?-log(uniform01(rng)):inc)/rate;
        while(j<nt-1){
            double dt=tk[j+1]-now;
            double slope=(acid[j+1]-acid[j])/(tk[j+1]-tk[j]);
            double c=acid[j]+slope*(now-tk[j]);
            double available=dt*(c+.5*slope*dt);
            bool hit=exposure<=available;
            double h=hit?2*exposure/(c+sqrt(fmax(0.,c*c+2*slope*exposure))):dt;
            if(death<=h*loss){now+=death/loss;status=-1;goto finish;}
            now+=h;death-=h*loss;
            if(hit){u+=inc;break;}
            exposure-=available;now=tk[++j];
        }
    }
    status=(u>=end);
finish:
    result[3*i]=now;result[3*i+1]=status;result[3*i+2]=u;
}
extern "C" int clusters(int n,int nt,int discrete,double du,double start,double end,void** p){
    track<<<(n+127)/128,128>>>(n,nt,discrete,du,start,end,(double*)p[0],(double*)p[1],
      (double*)p[2],(double*)p[3],(double*)p[4],(int*)p[5],(uint64_t*)p[6],(double*)p[7]);
    auto issued=cudaGetLastError();
    return issued==cudaSuccess?(int)cudaDeviceSynchronize():(int)issued;
}
