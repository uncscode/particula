// Forward-mode automatic differentiation through the gas, formation, transport,
// coagulation and two-region exchange. Each block carries one tangent direction.
#include <cuda_runtime.h>
#include <cmath>
struct D {double v,g;};
__device__ inline D val(double x){return {x,0};}
__device__ inline D operator+(D a,D b){return {a.v+b.v,a.g+b.g};}
__device__ inline D operator-(D a,D b){return {a.v-b.v,a.g-b.g};}
__device__ inline D operator*(D a,D b){return {a.v*b.v,a.g*b.v+a.v*b.g};}
__device__ inline D operator*(D a,double b){return {a.v*b,a.g*b};}
__device__ inline D operator*(double b,D a){return a*b;}
__device__ inline D operator/(D a,D b){return {a.v/b.v,(a.g*b.v-a.v*b.g)/(b.v*b.v)};}
__device__ inline D expd(D a){double v=exp(a.v);return {v,v*a.g};}
__device__ inline D slope(D l,D r){if(l.v*r.v<=0)return val(0);return fabs(l.v)<fabs(r.v)?l:r;}
#ifndef CLUSTER_CELLS
#define CLUSTER_CELLS 6
#endif
const int NC=CLUSTER_CELLS, NP=6;
__global__ __launch_bounds__(576) void advance(int n,int nt,int nd,int two,int model_entry,double dx,double dtmax,
 const double *input,const double *init,const double *v,const double *cap,const double *wall,
 const double *K,const int *offset,const int *left,const int *right,const double *weight,
 const double *cldata,const double *param,double tau,double *out
#ifdef MULTIBOX
 ,const unsigned long long *boxes
#endif
 ){
    int dir=blockIdx.x%nd,b=blockIdx.x/nd,i=threadIdx.x%n,z=threadIdx.x/n,j=threadIdx.x,other=(1-z)*n+i;
#ifdef MULTIBOX
    const unsigned long long *box=boxes+15*b;
    input=(const double*)box[0];init=(const double*)box[1];v=(const double*)box[2];cap=(const double*)box[3];wall=(const double*)box[4];K=(const double*)box[5];
    offset=(const int*)box[6];left=(const int*)box[7];right=(const int*)box[8];weight=(const double*)box[9];cldata=(const double*)box[10];
    two=(int)box[11];model_entry=(int)box[12];tau=(double)box[13];
#endif
    int N=2*n;extern __shared__ D shared[];
    D *pop=shared,*flux=pop+N,*save=flux+N,*tmp=save+N;
    __shared__ D acid[2],clusters[2][NC],inlet[2],cs[2];
    #ifdef AUDIT
    double particle_budget=0,entry_total=0,gas_net=0,gas_supply=0,cluster_net=0,cluster_supply=0;
#endif
    D p[NP];for(int a=0;a<NP;a++)p[a]={param[b*NP+a],(dir==a+1)?1.:0.};
    D mix=p[0],ws=expd(p[1]),prod=expd(p[2]),gaswall=expd(p[3]),kn=expd(p[4]),ki=expd(p[5]);
    D speed=val(v[i+1])+mix*(v[n+1+i+1]-v[i+1]);
    D inlet_speed=val(v[0])+mix*(v[n+1]-v[0]);
    D capture=val(cap[i])+mix*(cap[n+i]-cap[i]);
    pop[j]=val(init[i]);if(i==0){acid[z]=val(input[3]);cs[z]=val(0);inlet[z]=val(0);
      for(int c=0;c<NC;c++)clusters[z][c]=val(0);}
    __syncthreads();
    for(int it=0;it<nt;it++){
        // A single mean observation operator, plus the illuminated-zone gas diagnostic.
        int pos=((b*nd+dir)*nt+it)*(n+NC+4);
        if(z==0){
            D mean=(pop[i]+pop[other])*.5;out[pos+i]=dir==0?mean.v:mean.g;
            if(i==0){
                for(int c=0;c<NC;c++){D x=(clusters[0][c]+clusters[1][c])*.5;out[pos+n+c]=dir?x.g:x.v;}
                D gas=(acid[0]+acid[1])*.5,ent=(inlet[0]+inlet[1])*.5,sink=(cs[0]+cs[1])*.5;
                out[pos+n+NC]=dir?gas.g:gas.v;out[pos+n+NC+1]=dir?ent.g:ent.v;
                out[pos+n+NC+2]=dir?sink.g:sink.v;out[pos+n+NC+3]=dir?acid[0].g:acid[0].v;
            }
        }
        __syncthreads();if(it==nt-1)break;
        const double *f=input+it*6,*next=f+6;double dur=(next[0]-f[0])*3600;
        // Uniform CFL schedule across derivative evaluations and parameter values.
        double upper=fmax(f[5],next[5]);int ns=max(1,(int)ceil(dur/dtmax));
        ns=max(ns,(int)ceil(dur*upper*v[2*(n+1)]/(.42*dx)));
        double h=dur/ns,exchange=two?(1-exp(-h/tau))*.5:0;
        for(int s=0;s<ns;s++){
            double a=(s+.5)/ns, production=(1-a)*f[1]+a*next[1];
            double entry=(1-a)*f[2]+a*next[2];int flags=(int)(a<.5?f[4]:next[4]);
            tmp[j]=capture*pop[j];__syncthreads();
            if(i==0){
                D sink=val(0);for(int k=0;k<n;k++)sink=sink+tmp[z*n+k];
                if(model_entry)for(int c=0;c<NC;c++)sink=sink+clusters[z][c]*(val(cldata[3*NC+c])+mix*(cldata[4*NC+c]-cldata[3*NC+c]));
                cs[z]=sink; D loss=gaswall*.002+val(.0001754037)+sink;
                double illumination=two && (flags&2)?(z==0?2.:0.):1.;
                D source=prod*(fmax(0.,production-150.103)*illumination)+val(150.103);
                #ifdef AUDIT
                double gas_old=acid[z].v;
#endif
                D frac=expd(loss*(-.5*h));D mid=acid[z]*frac+source/loss*(val(1)-frac);
                acid[z]=mid*frac+source/loss*(val(1)-frac);
                #ifdef AUDIT
                gas_net+=acid[z].v-gas_old;gas_supply+=source.v*h;
#endif
                tmp[z*n]=mid;
                if(model_entry){
                    D norm=mid*(1./6e7),cube=norm*norm*norm;
                    D neutral=kn*cube*16.59,ion=ki*cube*132.2;
                    D birth=neutral+((flags&1)?ion*7./(ion+val(7)):val(0));
                    // Six implicit transfer cells from the published 0.8 nm birth to 2.9 nm.
                    #ifdef AUDIT
                    cluster_supply+=birth.v*h;cluster_net+=birth.v*h;
#endif
                    for(int c=0;c<NC;c++){
                        D rate=(val(cldata[c])+mix*(cldata[NC+c]-cldata[c]))*mid;
                        D count=(clusters[z][c]+birth*h)/(val(1)+(rate+ws*cldata[2*NC+c]+val(.0001754037))*h);
                        clusters[z][c]=count;birth=rate*count;
#ifdef AUDIT
                        cluster_net-=h*(ws.v*cldata[2*NC+c]+.0001754037)*count.v;
#endif
                    }
                    inlet[z]=birth;
#ifdef AUDIT
                    cluster_net-=birth.v*h;
#endif
                }else inlet[z]=val(entry);
            }
            __syncthreads();D mid=tmp[z*n];
            save[j]=pop[j];__syncthreads();
            for(int rk=0;rk<2;rk++){
                D grad=val(0);if(i>0&&i<n-1)grad=slope(pop[j]-pop[j-1],pop[j+1]-pop[j]);
                flux[j]=speed*mid*(pop[j]+grad*.5)*(1./dx);__syncthreads();
                D incoming=i?flux[j-1]:inlet[z];
                #ifdef AUDIT
                double admitted=i==0?inlet[z].v:0.;
                entry_total+=.5*h*admitted;
                particle_budget+=.5*h*(admitted-(i==n-1?flux[j].v:0.)-(ws.v*wall[i]+.0001754037)*pop[j].v);
#endif
                tmp[j]=incoming-flux[j]-(ws*wall[i]+val(.0001754037))*pop[j];__syncthreads();
                pop[j]=rk?(save[j]+pop[j]+h*tmp[j])*.5:save[j]+h*tmp[j];__syncthreads();
            }
            D gain=val(0),haz=val(0);
            for(int k=0;k<n;k++){
                haz=haz+pop[z*n+k]*K[i*n+k];
#ifdef AUDIT
                particle_budget-=.5*h*K[i*n+k]*pop[j].v*pop[z*n+k].v*(1.+(k>=(int)wall[n+i]));
#endif
            }
            for(int k=offset[i];k<offset[i+1];k++)gain=gain+pop[z*n+left[k]]*pop[z*n+right[k]]*weight[k];
            tmp[j]=pop[j]+(gain-haz*pop[j])*h;__syncthreads();
            pop[j]=tmp[j]+(tmp[other]-tmp[j])*exchange;
            if(i==0)save[j]=acid[z];
            if(i<NC)flux[j]=clusters[z][i];
            __syncthreads();
            if(i==0)acid[z]=save[j]+(save[other]-save[j])*exchange;
            if(i<NC)clusters[z][i]=flux[j]+(flux[other]-flux[j])*exchange;
            __syncthreads();
        }
    }
#ifdef AUDIT
    tmp[j]=val(pop[j].v-init[i]-particle_budget);flux[j]=val(entry_total+init[i]);
    save[j]=val(0);__syncthreads();
    if(i==0){
        double sum=0;for(int c=0;c<NC;c++)sum+=clusters[z][c].v;
        save[j]=val(sum-cluster_net);save[j+1]=val(cluster_supply);
        save[j+2]=val(acid[z].v-input[3]-gas_net);save[j+3]=val(gas_supply+input[3]);
    }
    __syncthreads();
    if(j==0){
        double err=0,scale=0;for(int h=0;h<N;h++){err+=tmp[h].v;scale+=flux[h].v;}
        int end=((b*nd+dir)*nt+nt-1)*(n+NC+4)+n+NC;
        out[end]=fabs(err)/fmax(1.,scale);
        out[end+1]=fabs(save[0].v+save[n].v)/fmax(1.,save[1].v+save[n+1].v);
        out[end+2]=fabs(save[2].v+save[n+2].v)/fmax(1.,save[3].v+save[n+3].v);
    }
#endif
}
#ifndef MULTIBOX
extern "C" int inverse_run(int count,int n,int nt,int nd,int two,int entry,double dx,double dt,double tau,void **a){
    advance<<<count*nd,2*n,8*n*sizeof(D)>>>(n,nt,nd,two,entry,dx,dt,(double*)a[0],(double*)a[1],(double*)a[2],(double*)a[3],(double*)a[4],(double*)a[5],(int*)a[6],(int*)a[7],(int*)a[8],(double*)a[9],(double*)a[10],(double*)a[11],tau,(double*)a[12]);
    cudaError_t issued=cudaGetLastError();
    return (int)(issued==cudaSuccess?cudaDeviceSynchronize():issued);
}
#else
extern "C" int many_run(int count,int n,int nt,double dx,double dt,void* boxes,void* params,void* out){
    advance<<<count,2*n,8*n*sizeof(D)>>>(n,nt,1,0,0,dx,dt,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,(const double*)params,80.,(double*)out,(const unsigned long long*)boxes);
    cudaError_t issued=cudaGetLastError();
    return (int)(issued==cudaSuccess?cudaDeviceSynchronize():issued);
}
#endif
