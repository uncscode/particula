"""Sectional water inventory, diameter mapping, and directly evaluated AIOMFAC."""
import os
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import particula as par
import particula_aiomfac as backend
import study
from study import T, P, overlap, acid_pair

PACKET=Path(os.environ["PACKET"])
assert Path(backend.__file__).resolve().is_relative_to(Path(os.environ["PARTICULA_ROOT"]))
Z=np.load(PACKET/"spectra.npz")
t,d,e,y,stage=(Z[k] for k in ["time","diameter","edges","observed","stage"])
frame=pd.read_csv(PACKET/"lamp_budget.csv")
missing=frame.small_missing.to_numpy()
segments={"train":np.flatnonzero(stage<=7),"test":np.flatnonzero((stage>=9)&(stage<=18))}
g0=1.1
w0=.5
kelvin=4*.079*.01801528/(1000*8.314462618*T)*1e9
amu=1.66053906660e-27
salt=98.079+2*17.03052
groups={"water":{16:1},"ammonium sulfate":{204:2,261:1}}
model=par.particles.ActivityFactory().get_strategy("aiomfac",{
    "mixture":backend.MixtureSpec(groups),"material_mapping":dict(enumerate(groups)),
    "phase_mode":"single_liquid"})

@lru_cache(maxsize=None)
def water(D):
    target=.58*np.exp(-kelvin/D)
    mass=1350*np.pi/6*(D*1e-9)**3
    def residual(f):
        return float(model.activities_at_temperature(mass*np.array([f,1-f]),T,["water"])[0])-target
    lo=.005 if D<2.9 else .07
    f=brentq(residual,lo,.75,xtol=1e-12)
    assert abs(residual(f))<1e-9
    return f

def fraction(D,law):
    if law=="acidic":
        from acid_thermo import fraction as speciated
        return speciated(D)
    if law=="dry":return np.zeros_like(D)
    if law=="original":
        x=.58*np.exp(-kelvin/D)
        return 1-salt/(salt+3*18.01528*x/(1-x))
    return np.array([water(float(x)) for x in D])

def solid_mass(D,law):
    rho=1770 if law=="dry" else 1350
    return rho*np.pi/6*(D*1e-9)**3*(1-fraction(D,law))

def reported(D,law,retain):
    dry3=solid_mass(D,law)/(1770*np.pi/6)*1e27
    return (dry3+retain*(D**3-dry3))**(1/3)

def capture(D):
    diff=8e-6*(T/293.15)**1.75
    r=D*.5e-9
    speed=np.sqrt(8*8.314462618*T/(np.pi*.098079))
    vapor=par.dynamics.get_first_order_mass_transport_k(r,par.particles.get_vapor_transition_correction(3*diff/(speed*r),1.),diff)
    blend=np.clip(np.log(19.5/D)/np.log(19.5/9.7),0,1)
    pair=[]
    for ham in (0.,6.4e-20):
        ref,k=acid_pair(T,P,ham,1.,1500.,19.5,151)
        pair.append(1e6*(vapor+blend*(np.interp(D,ref,k)-vapor)))
    return np.array(pair)

@lru_cache(maxsize=None)
def geometry(n,law="original",retain=1.,legacy=False):
    edges=np.linspace(np.log10(2.9),e[-1] if legacy else np.log10(96.),n+1)
    dx=edges[1]-edges[0]; D=10**edges;mid=10**((edges[1:]+edges[:-1])/2)
    salt_mw=98.079+1.5*17.03052 if law=="acidic" else salt
    der=(solid_mass(D*(1+2e-5),law)-solid_mass(D*(1-2e-5),law))/(4e-5*D*salt_mw*amu)
    cap=capture(D); vel=cap/der/(D*np.log(10))
    if legacy:
        dx,edges,base,wall,K,pairs,over,mi,mo=study.grid(n,d,e)
        return dict(edges=edges,D=mid,dx=dx,vel=base[[2,1]],wall=wall,K=K,pairs=pairs,over=over,
                    mi=mi,mo=mo,cap=capture(mid),mapped=edges)
    wall=.002*np.sqrt(par.particles.get_diffusion_coefficient_via_system_state(mid*.5e-9,T,P)/(8e-6*(T/293.15)**1.75))
    m=solid_mass(mid,law);physical_mass=(1770 if law=="dry" else 1350)*np.pi/6*(mid*1e-9)**3
    K=par.dynamics.get_brownian_kernel_via_system_state(mid*.5e-9,physical_mass,T,P)*1e6
    i,j=np.triu_indices(n); total=m[i]+m[j];hi=np.searchsorted(m,total);ok=hi<n
    i,j,total,hi=(a[ok] for a in (i,j,total,hi)); f=(total-m[hi-1])/(m[hi]-m[hi-1])
    k=K[i,j]*np.where(i==j,.5,1);dest=np.r_[hi,hi-1];order=np.argsort(dest)
    pairs=[np.r_[0,np.bincount(dest,minlength=n).cumsum()].astype('i4'),np.tile(i,2)[order].astype('i4'),
           np.tile(j,2)[order].astype('i4'),np.r_[k*f,k*(1-f)][order]]
    over=np.searchsorted(m,m[-1]-m,side="right").astype('i4')
    # Audit coagulation on the conserved sulfate mass coordinate.
    probe=np.exp(-((np.log(mid/12))/.55)**2)
    pair=K*probe[:,None]*probe; excess=m[:,None]+m>m[-1]
    gain=np.bincount(dest[order],weights=pairs[3]*probe[pairs[1]]*probe[pairs[2]],minlength=n)
    assert abs((gain-pair.sum(1))@m+.5*np.sum(pair*excess*(m[:,None]+m)))<1e-10*(pair.sum(1)@m)
    mapped=np.log10(reported(D,law,retain))
    mi=overlap(mapped[:-1],mapped[1:],e)
    mo=overlap(e[:-1],e[1:],mapped)/np.diff(mapped)
    assert np.all(mo.sum(0)<=1+1e-11) and np.all(np.diff(mapped)>0)
    return dict(edges=edges,D=mid,dx=dx,vel=vel,wall=wall,K=K,pairs=pairs,over=over,
                mi=mi,mo=mo,cap=capture(mid),mapped=mapped)

def common_input():
    vel=geometry(64,legacy=True)["vel"][1,0]
    return g0*vel*frame.measured_cal.to_numpy()*y[:,3]

def metrics(spec,idx):
    w=np.diff(e); obs=y[idx]; widths=w[4:]
    err=abs(np.log10(1+spec[:,4:])-np.log10(1+obs[:,4:]))@widths/widths.sum()
    cnt=spec[:,4:]@widths;actual=obs[:,4:]@widths
    tail=np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))
    curve=spec@tail
    def first(q):
        hits=np.flatnonzero(curve>=q)
        return float(t[idx[hits[0]]]) if len(hits) else np.nan
    rows=[]
    for st in [0,*np.unique(stage[idx])]:
        mask=(np.arange(len(idx))>0)&((stage[idx]<18) if st==0 else (stage[idx]==st))
        if not mask.any():continue
        vals=spec[mask,4:];weights=vals*widths;ld=np.log(d[4:]);gm=(weights@ld)/np.maximum(weights.sum(1),1e-30)
        sd=np.exp(np.sqrt(np.sum(weights*(ld-gm[:,None])**2,axis=1)/np.maximum(weights.sum(1),1e-30)))
        rows.append(dict(stage=int(st),log_error=float(err[mask].mean()),factor=10**err[mask].mean(),
          count=cnt[mask].mean(),observed_count=actual[mask].mean(),mode_nm=np.median(d[4:][vals.argmax(1)]),
          width=np.median(sd),tail=curve[mask].mean(),observed_tail=(obs[mask]@tail).mean(),
          first1000=first(1000),first5000=first(5000)))
    return rows
