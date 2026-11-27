// One independent seven-bin evaporation trajectory per GPU thread.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

struct Conditions {
    double initial_radius, initial_mass, water_ratio, organic_share;
    double mean_free_path, gas_diffusivity, kelvin_length, accommodation;
};

__device__ void timescales(double total, Conditions s, double inverse_d0,
                           double sensitivity, double &gas, double &supply) {
    const double pi = 3.14159265358979323846;
    const double r = s.initial_radius * cbrt(total * (1. + s.water_ratio));
    const double kn = s.mean_free_path / r;
    const double a = s.accommodation;
    const double beta = .75*a*(1.+kn)/(kn*kn+kn+.283*a*kn+.75*a);
    const double k = 4.*pi*r*s.gas_diffusivity*beta;
    gas = s.initial_mass * total / (k*exp(s.kelvin_length/r)*s.organic_share*1e-9);
    supply = (r*r/(pi*pi))*inverse_d0*pow(10.,sensitivity*(1.-total));
}

__global__ void integrate(Conditions s, const double *fractions, const double *params,
        const int count, const double *dt, const int *events, const int steps,
        const int outputs, double *result) {
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= count) return;
    const double inverse_d0=params[3*id], g=params[3*id+1], extra=params[3*id+2];
    double m[7];
    const double inverse_c[7]={1e4,1e3,1e2,10.,1.,.1,.01};
    double total=0.;
    for (int j=0;j<7;++j) {
        m[j]=fractions[j]*(1.-extra)+(j==0?extra:0.);
        total+=m[j];
    }
    for (int i=0;i<steps;++i) {
        double gas,supply;
        timescales(total,s,inverse_d0,g,gas,supply);
        double midpoint=0.;
        for (int j=0;j<7;++j)
            midpoint+=m[j]*exp(-.5*dt[i]/(gas*inverse_c[j]+supply));
        timescales(midpoint,s,inverse_d0,g,gas,supply);
        total=0.;
        for (int j=0;j<7;++j) {
            m[j]*=exp(-dt[i]/(gas*inverse_c[j]+supply)); total+=m[j];
        }
        if (events[i]>=0) result[id*outputs+events[i]]=total;
    }
}

// The Python entry point supplies contiguous arrays; synchronize before returning.
extern "C" int solve(const double *conditions,const double *fractions,const double *params,
        int count,const double *dt,const int *events,int steps,int outputs,double *result) {
    Conditions s={conditions[0],conditions[1],conditions[2],conditions[3],
                  conditions[4],conditions[5],conditions[6],conditions[7]};
    double *device=nullptr; int *indices=nullptr;
    const size_t bytes=(7+3*size_t(count)+steps+size_t(count)*outputs)*sizeof(double);
    cudaError_t err=cudaMalloc(&device,bytes);
    if(err!=cudaSuccess) return err;
    double *p=device+7,*d=p+3*count,*y=d+steps;
    err=cudaMalloc(&indices,sizeof(int)*steps);
    if(err==cudaSuccess) err=cudaMemcpy(device,fractions,7*sizeof(double),cudaMemcpyHostToDevice);
    if(err==cudaSuccess) err=cudaMemcpy(p,params,3*count*sizeof(double),cudaMemcpyHostToDevice);
    if(err==cudaSuccess) err=cudaMemcpy(d,dt,steps*sizeof(double),cudaMemcpyHostToDevice);
    if(err==cudaSuccess) err=cudaMemcpy(indices,events,steps*sizeof(int),cudaMemcpyHostToDevice);
    if(err==cudaSuccess) {
        integrate<<<(count+127)/128,128>>>(s,device,p,count,d,indices,steps,outputs,y);
        err=cudaGetLastError();
    }
    if(err==cudaSuccess) err=cudaMemcpy(result,y,size_t(count)*outputs*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(indices); cudaFree(device);
    return err;
}
