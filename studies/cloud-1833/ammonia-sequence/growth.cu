// Event-driven sulfate and signed charge for an independent cluster.
// The acid interpolation is linear, all other hazards constant within half a sample.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
__device__ double uniform01(uint64_t &s) {
    s^=s>>12; s^=s<<25; s^=s>>27;
    return ((s*2685821657736338717ULL >> 11)+0.5)*1.1102230246251565e-16;
}
__global__ void track(int count,int nt,int track_charge,int absorb,double start,double end,
 const double* time,const double* acid,const double* physics,const double* ion,const double* evaporation,
 const double* birth,const int* first,const int* birth_charge,const uint64_t* seeds,double* result,int* counts){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    uint64_t rng=seeds[i];
    int s=0,j=first[i],q=track_charge?birth_charge[i]:0,status=0,up=0,down=0,att=0,neutral=0;
    double now=birth[i],passed17=-1.;
    while(start+s<end && j<nt-1){
        const double *p=&physics[8*s];
        double mid=(time[j]+time[j+1])*.5;
        int half=(now>=mid),k=j+half;
        double stop=half?time[j+1]:mid;
        double slope=(acid[j+1]-acid[j])/(time[j+1]-time[j]);
        double capture=q<0?p[6]:(q>0?p[7]:p[0]);
        double grow=capture*(acid[j]+slope*(now-time[j]));
        const double *a=&ion[3*k];
        double neg=0,pos=0;
        if(track_charge){
            if(q==0){neg=p[2]*a[0];pos=p[2]*a[1];}
            if(q>0)neg=p[3]*a[0];
            if(q<0)pos=p[4]*a[1];
        }
        double clear=abs(q)*p[5]*a[2],evap=evaporation[3*s+q+1];
        double total=grow+p[1]+neg+pos+clear+evap,delta=stop-now;
        double hazard=-log(uniform01(rng));
        if(hazard>=delta*(total+.5*capture*slope*delta)){
            now=stop;
            if(half){
                ++j;
                if(q && start+s>=end*pow(2.2/2.9,3.))
                    atomicAdd(counts+2*j+(q>0),1);
            }
            continue;
        }
        double h=2*hazard/(total+sqrt(fmax(0.,total*total+2*capture*slope*hazard)));
        now+=h;grow+=capture*slope*h;
        double pick=uniform01(rng)*(grow+p[1]+neg+pos+clear+evap);
        if((pick-=grow)<0){++s;++up;if(passed17<0 && start+s>=end*pow(1.7/2.9,3.))passed17=now;continue;}
        if((pick-=p[1])<0){status=-1;break;}
        if((pick-=neg)<0){if(q)++neutral;else ++att;--q;continue;}
        if((pick-=pos)<0){if(q)++neutral;else ++att;++q;continue;}
        if((pick-=clear)<0){status=-2;break;}
        --s;++down;
        if(s<0 || (absorb && s==0)){status=-3;break;}
    }
    if(start+s>=end)status=1;
    double* out=result+i*8;
    out[0]=now;out[1]=status;out[2]=start+s;out[3]=q;
    out[4]=up;out[5]=down;out[6]=att;out[7]=passed17;
}
extern "C" int clusters(int n,int nt,int charge,int absorb,double start,double end,void** p){
    track<<<(n+127)/128,128>>>(n,nt,charge,absorb,start,end,(double*)p[0],(double*)p[1],
        (double*)p[2],(double*)p[3],(double*)p[4],(double*)p[5],(int*)p[6],(int*)p[7],
        (uint64_t*)p[8],(double*)p[9],(int*)p[10]);
    auto status=cudaGetLastError();
    return status==cudaSuccess?(int)cudaDeviceSynchronize():(int)status;
}
