// One well-mixed chamber per block, one logarithmic size cell per thread.
#include <cuda_runtime.h>
#include <cmath>

__global__ void chambers(int bins,int nt,double dx,double cap,
 const double *hours,const double *gas,const double *inlet,const double *init,
 const double *velocity,const double *wall,const double *setting,
 const double *kernel,const int *offset,const int *left,const int *right,const double *weight,
 const int *first_overflow,const double *clear,const double *flux_ref,double *output,double *ledger,const double *tag_in,double *tag_out,double *tag_book){
    extern __shared__ double shared[];
    double *n=shared,*flux=n+bins,*base=flux+bins,*change=base+bins;
    double *tag=change+bins,*tag_flux=tag+bins,*tag_base=tag_flux+bins,*tag_change=tag_base+bins;
    int i=threadIdx.x,c=blockIdx.x;
    double scale=setting[6*c], loss=setting[6*c+1];
    int coal=(int)setting[6*c+3];
    const double *v=velocity+c*(bins+2), *acid=gas+c*nt;
    double ledger_in=0,ledger_out=0,ledger_wall=0,ledger_dil=0,ledger_coag=0,ledger_clear=0;
    double tag_supplied=0, tag_removed=0;
    tag[i]=0; tag_out[(c*nt)*bins+i]=0;
    n[i]=init[i]; output[(c*nt)*bins+i]=n[i]; __syncthreads();
    for(int t=0;t<nt-1;t++){
        double dur=(hours[t+1]-hours[t])*3600;
        int ns=max(1,(int)ceil(dur/cap));
        ns=max(ns,(int)ceil(dur*fmax(acid[t],acid[t+1])*scale*v[bins+1]/(0.42*dx)));
        double h=dur/ns;
        for(int s=0;s<ns;s++){
            double f=(s+0.5)/ns, a=scale*((1-f)*acid[t]+f*acid[t+1]);
            double entry=a*v[0]*((1-f)*inlet[t]+f*inlet[t+1]);
            if(flux_ref[c*nt+t]>=0) entry=((1-f)*flux_ref[c*nt+t]+f*flux_ref[c*nt+t+1])*((1-f)*inlet[t]+f*inlet[t+1]);
            if(setting[6*c+2]>0) entry=flux_ref[c*nt+t];
            double injected=(1-f)*tag_in[c*nt+t]+f*tag_in[c*nt+t+1];
            tag_base[i]=tag[i];base[i]=n[i]; __syncthreads();
            for(int rk=0;rk<2;rk++){
                double slope=0;
                if(i>0 && i<bins-1){
                    double l=n[i]-n[i-1],r=n[i+1]-n[i];
                    if(l*r>0) slope=copysign(fmin(fabs(l),fabs(r)),l);
                }
                flux[i]=a*v[i+1]*(n[i]+0.5*slope)/dx;
                __syncthreads();
                tag_flux[i]=n[i]>0 ? flux[i]*tag[i]/n[i] : 0;
                __syncthreads();
                double tag_incoming=i==0?injected:tag_flux[i-1];
                double tag_loss=(loss*wall[c*bins+i]+setting[6*c+4])*tag[i];
                tag_change[i]=tag_incoming-tag_flux[i]-tag_loss;
                tag_supplied+=i==0?0.5*h*injected:0;
                tag_removed+=0.5*h*(tag_loss+(i==bins-1?tag_flux[i]:0));
                double incoming=i==0?entry:flux[i-1];
                change[i]=incoming-flux[i]-(loss*wall[c*bins+i]+setting[6*c+4])*n[i];
                ledger_in+=i==0?0.5*h*entry:0;
                ledger_out+=i==bins-1?0.5*h*flux[i]:0;
                ledger_wall+=0.5*h*loss*wall[c*bins+i]*n[i];
                ledger_dil+=0.5*h*setting[6*c+4]*n[i];
                __syncthreads();
                tag[i]=rk==0?tag_base[i]+h*tag_change[i]:0.5*(tag_base[i]+tag[i]+h*tag_change[i]);
                n[i]=rk==0?base[i]+h*change[i]:0.5*(base[i]+n[i]+h*change[i]);
                __syncthreads();
            }
            if(coal){
                double gain=0, sink=0,overflow=0,tgain=0,tsink=0,tover=0;
                for(int j=0;j<bins;j++){
                    double pair=kernel[i*bins+j]*n[j]*n[i];
                    double loss_tag=kernel[i*bins+j]*tag[i]*n[j];
                    tsink+=loss_tag; if(j>=first_overflow[i]) tover+=loss_tag;
                    sink+=pair;
                    if(j>=first_overflow[i]) overflow+=pair;
                }
                for(int p=offset[i];p<offset[i+1];p++)
                    {
                    gain+=weight[p]*n[left[p]]*n[right[p]];
                    tgain+=weight[p]*(tag[left[p]]*n[right[p]]+n[left[p]]*tag[right[p]]);
                }
                tag_change[i]=h*setting[6*c+5]*(tgain-tsink);
                tag_removed+=h*setting[6*c+5]*tover;
                ledger_coag+=0.5*h*setting[6*c+5]*(sink+overflow);
                change[i]=h*setting[6*c+5]*(gain-sink); __syncthreads();
                tag[i]+=tag_change[i];n[i]+=change[i]; __syncthreads();
            }
        }
        tag_removed+=clear[t+1]*tag[i];tag[i]*=(1-clear[t+1]);
        tag_out[(c*nt+t+1)*bins+i]=tag[i];
        ledger_clear+=clear[t+1]*n[i]; n[i]*=(1-clear[t+1]);
        output[(c*nt+t+1)*bins+i]=n[i]; __syncthreads();
    }
    double *l=ledger+(c*bins+i)*6;
    tag_book[(c*bins+i)*2]=tag_supplied;tag_book[(c*bins+i)*2+1]=tag_removed;
    l[0]=ledger_in;l[1]=ledger_out;l[2]=ledger_wall;l[3]=ledger_dil;l[4]=ledger_coag;l[5]=ledger_clear;
}

extern "C" int replay(int cases,int bins,int nt,double dx,double cap,void **a){
    chambers<<<cases,bins,8*bins*sizeof(double)>>>(bins,nt,dx,cap,
        (double*)a[0],(double*)a[1],(double*)a[2],(double*)a[3],(double*)a[4],
        (double*)a[5],(double*)a[6],(double*)a[7],(int*)a[8],(int*)a[9],(int*)a[10],
        (double*)a[11],(int*)a[12],(double*)a[13],(double*)a[14],(double*)a[15],(double*)a[16],(double*)a[17],(double*)a[18],(double*)a[19]);
    return (int)cudaDeviceSynchronize();
}
