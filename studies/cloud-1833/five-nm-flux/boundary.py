"""Infer the 5 nm number flux without using any measurement above the 5.9 nm channel."""
from physics import PACKET,segments,t,stage,d,e,y,missing,solid_mass,capture,amu,salt,par,T,P
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from pathlib import Path

ix=segments['test'];hours=t[ix];seconds=(hours-hours[0])*3600;st=stage[ix];obs=y[ix];gap=missing[ix]
acid=pd.read_csv(PACKET/'forcing.csv').measured_cal.to_numpy()[ix]
fit=pd.read_csv(PACKET/'inverse-model/frozen-final.csv').query('k==1 and entry==1').iloc[0]
blend=fit.collision_fraction
wall_scale=np.exp(fit.log_particle_wall)
def growth(D):
    deriv=(solid_mass(D*(1+2e-5),'aiomfac')-solid_mass(D*(1-2e-5),'aiomfac'))/(4e-5*D*salt*amu)
    return 1.1*capture(D)/(deriv*D*np.log(10))
def wall(D):
    return .002*np.sqrt(par.particles.get_diffusion_coefficient_via_system_state(D*.5e-9,T,P)/(8e-6*(T/293.15)**1.75))
def brown(D):
    return par.dynamics.get_brownian_kernel_via_system_state(D*.5e-9,1350*np.pi/6*(D*1e-9)**3,T,P)*1e6
def reconstruct():
    centers=np.array([d[4],5.,d[5]])
    hard,attractive=growth(centers)
    v=hard+blend*(attractive-hard)
    weight=np.log(5/d[4])/np.log(d[5]/d[4])
    densities=obs[:,[4,5]].copy()
    assert np.all(np.flatnonzero(gap)==np.arange(np.flatnonzero(gap)[0],len(gap)))
    last=np.flatnonzero(~gap)[-1]
    densities[gap]=densities[last]
    density=densities@np.array([1-weight,weight])
    direct=density*acid*v[1]
    out=pd.DataFrame(dict(archive_h=hours,stage=st,small_channel_gap=gap,acid_cm3=acid,density4=densities[:,0],density5p9=densities[:,1],
                          direct_hard=density*acid*hard[1],direct=direct,direct_attractive=density*acid*attractive[1]))
    # Estimate first-order loss of a tagged 5 nm particle. This small upstream
    # correction uses only the two input channels; large-particle sink tested separately.
    K=brown(centers)
    removal=wall_scale*wall(np.array([5.]))[0]+.0001754037+(densities*np.diff(e)[[4,5]])@K[1,[0,2]]
    integrated_loss=cumulative_trapezoid(removal,seconds,initial=0.)
    exposure=cumulative_trapezoid(acid,seconds,initial=0.)
    for k,name,idx in [(0,'from4',0),(2,'from5p9',1)]:
        D=np.geomspace(5.,centers[k],65);a,b=growth(D)
        travel=np.trapezoid(1/(a+blend*(b-a)),np.log10(D))
        anchor=np.interp(exposure+travel,exposure,seconds,left=np.nan,right=np.nan)
        valid=np.isfinite(anchor)&(np.abs(anchor-seconds)<=1800)&(anchor<=seconds[last])&~gap
        age=anchor-seconds
        factor=np.exp(np.interp(anchor,seconds,integrated_loss)-integrated_loss)
        estimate=acid*v[k]*np.interp(anchor,seconds,densities[:,idx])*factor
        out[name]=np.where(valid,estimate,direct)
        out[name+'_valid']=valid
        out[name+'_anchor_lag_minutes']=np.where(valid,age/60,np.nan)
    # Finite interval number balance over [5.0, 5.9] nm. The derivative
    # tests transient accumulation independently of propagating either endpoint.
    from scipy.ndimage import convolve1d
    inventory=.5*(density+densities[:,1])*np.log10(centers[2]/5.)
    sinks=acid*v[2]*densities[:,1]+removal*inventory
    for name,length in [('balance',3),('balance_wide',5)]:
        weights=np.r_[np.arange(1,length+1),np.arange(length-1,0,-1)]
        smooth=lambda u:convolve1d(u,weights/weights.sum(),mode='nearest')
        raw=np.gradient(smooth(inventory),seconds)+smooth(sinks)
        out[name+'_raw']=raw
        out[name]=np.where(gap,direct,np.maximum(0.,raw))
    # Three assumptions during the late-dark small-instrument outage.
    for label,f in [('zero',0.),('max',np.max(density[(st==18)&~gap]))]:
        out['dark_'+label]=np.where(gap,acid*v[1]*f,direct)
    # Approximate two-subthreshold Brownian mergers that create a new >5 nm particle.
    lo=10**e[:-1];hi=np.minimum(10**e[1:],5.);ok=hi>lo
    D=np.sqrt(lo[ok]*hi[ok]);N=obs[:,ok]*np.log10(hi[ok]/lo[ok])
    N[gap]=N[last]
    K=brown(D);passed=(D[:,None]**3+D[None,:]**3>=5**3)
    out['coag_crossing']=.5*np.einsum('ti,ij,tj->t',N,K*passed,N)
    out['direct_plus_coag']=direct+out.coag_crossing
    return out
