"""Finite-state kinetic network for molecular exchange, with open source and exit."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import constants as c
from scipy.linalg import lu_factor,lu_solve
from evaporation_reference import acid_capture,hydrate_factor

ROOT=Path(__file__).resolve().parent
TK=298.15
STANDARD=c.atm/(c.k*TK)/1e6
RT=c.R*TK/4184
E={tuple(r[:3]):r[3] for r in pd.read_csv(ROOT/'energies.csv').itertuples(index=False,name=None)}
def ammonia_capture(n,m,q):
    acid=.098079/c.Avogadro;base=.01703052/c.Avogadro
    mass=n*acid+m*base+q*c.proton_mass
    mu=mass*base/(mass+base)
    if q:
        alpha=2.81e-30
        x=1.47*3.33564e-30/np.sqrt(8*np.pi*c.epsilon_0*alpha*c.k*TK)
        f=.4767*x+.6200 if x>=2 else (x+.509)**2/10.526+.9754
        return 1e6*f*c.e*np.sqrt(np.pi*alpha/(c.epsilon_0*mu))
    volume=n*acid/1830+m*base/696
    radius=lambda v:(3*v/(4*np.pi))**(1/3)
    return 1e6*np.pi*(radius(volume)+radius(base/696))**2*np.sqrt(8*c.k*TK/(np.pi*mu))

def simulate(cfg, time, stage, acid, ions, rates, source):
    boundary=cfg.get('boundary',4)
    states=[key for key in E if 1<=key[0]<=boundary]
    idx={v:i for i,v in enumerate(states)}
    n,m,q=np.array(states).T
    count=len(states)
    # Matrices have columns as origin states; their coefficients multiply
    # unit, acid, ammonia, negative ions, positive ions and field respectively.
    mats=np.zeros((6,count,count))
    loss=np.zeros((6,count));escape=np.zeros(count); kill=np.zeros(count)
    flux=np.zeros((4,6,count)) # events: acid gain/loss and ammonia gain/loss
    projected=np.zeros((2,count))
    factor=cfg.get('exchange_speed',1.)
    def channel(i,j,k,coef):
        mats[k,j,i]+=coef
        mats[k,i,i]-=coef
    def water_partition(s):
        # Consistent free-energy adjustment for covered, rapidly equilibrated neutral hydrates.
        return cfg.get('energy_shifts',{}).get(str(s),0.)
    for i,s in enumerate(states):
        a,b,z=s
        for is_base,k in ((False,1),(True,2)):
            nxt=(a+(not is_base),b+is_base,z)
            beta=ammonia_capture(*s)*factor if is_base else acid_capture(*s)*1e6
            if nxt in idx:
                j=idx[nxt]
                dg=E[nxt]-E[s]+water_partition(nxt)-water_partition(s)
                sym=.5 if s==(1,0,0) and not is_base else 1.
                gamma=sym*beta*STANDARD*np.exp(dg/RT)
                channel(i,j,k,beta);channel(j,i,0,gamma)
                fi=2 if is_base else 0
                flux[fi,k,i]+=beta;flux[fi+1,0,j]+=gamma
            elif not is_base and a==boundary and (z!=0 or b>=cfg.get('neutral_exit_N',0)):
                escape[i]=beta; mats[k,i,i]-=beta;flux[0,k,i]+=beta
            elif is_base:
                projected[0,i]=beta
        if cfg.get('open_dimer') and a==1:
            kill[i]=cfg.get('seed_loss',.02)
        loss[0,i]=rates[a,0]+kill[i]
        loss[5,i]=abs(z)*rates[a,4]*cfg.get('field_scale',1.)
        for direction,k in [(-1,3),(1,4)]:
            if abs(z+direction)>1:continue
            nxt=(a,b,z+direction)
            coef=(rates[a,1]*cfg.get('attach',1)) if not z else rates[a,2 if z>0 else 3]*cfg.get('neutralize',1)
            if nxt in idx:channel(i,idx[nxt],k,coef)
            elif cfg.get('charge_projection'):
                candidates=[x for x in states if x[0]==a and x[2]==z+direction]
                if candidates:
                    j=idx[min(candidates,key=lambda x:abs(x[1]-b))]
                    channel(i,j,k,coef)
        mats[:,i,i]-=loss[:,i]
    source_n=cfg.get('source_n',1)
    pos_seed=min((i for i,s in enumerate(states) if s[0]==source_n and s[2]==1),key=lambda i:states[i][1])
    seeds=[idx[(source_n,0,-1)],idx[(cfg.get('neutral_n',source_n),cfg.get('source_N',0),0)],pos_seed]
    y=np.zeros(count);eye=np.eye(count);streams=[];logs=[]
    init=np.zeros(count)
    regional=np.zeros(count*2)
    contrast=cfg.get('contrast',0.)
    coupling=np.eye(count)/(2*cfg.get('mix_s',80.))
    for j,dt in enumerate(np.diff(time)):
        sub=2*int(np.ceil(dt/(2*cfg.get('dt',15.))-.0001));h=dt/sub
        budget=np.zeros(8);composition=np.zeros(4);stores=y.sum(); initial_species=np.array([y@n,y@m]); external=np.zeros(2)
        for i in range(sub):
            f=(i+.5)/sub; k=j+(f>=.5)
            ac=(1-f)*acid[j]+f*acid[j+1]
            nh=cfg.get('ppb',2.)*4.801082538e10/2
            if 'nh_stage11' in cfg and stage[k]==11:nh*=cfg['nh_stage11']
            air=ions[k]
            weights=np.array([1.,ac,nh,air[0],air[1],air[2]])
            prod=source(ac,stage[k],air,cfg)*(1.-np.asarray(cfg.get('mature_share',0.)))
            if contrast:
                intensity=contrast if stage[k] in (9,10,11,12,14,15,17) else 0.
                both=[];inputs=[];weights_region=[];src=[]
                for sign in (-1.,1.):
                    w=weights.copy();w[1]=ac*(1+sign*intensity)
                    local=source(ac if cfg.get('mean_births') else w[1],stage[k],air,cfg)*.5*(1.-np.asarray(cfg.get('mature_share',0.)))
                    init.fill(0);init[seeds]=local;inputs.append(init.copy());src.append(local)
                    both.append(eye+h*coupling-h*np.einsum('k,kij->ij',w,mats));weights_region.append(w)
                block=np.block([[both[0],-h*coupling],[-h*coupling,both[1]]])
                regional=lu_solve(lu_factor(block,check_finite=False),regional+h*np.concatenate(inputs),check_finite=False)
                parts=regional.reshape(2,count);y=parts.sum(0);prod=np.sum(src,axis=0)
                emitted=sum(w[1]*escape*v for w,v in zip(weights_region,parts))
                removed=sum((w@loss)*v for w,v in zip(weights_region,parts))
                event_rates=sum(np.einsum('k,rki,i->r',w,flux,v) for w,v in zip(weights_region,parts))
            else:
                init.fill(0);init[seeds]=prod
                matrix=eye-h*np.einsum('k,kij->ij',weights,mats)
                y=lu_solve(lu_factor(matrix,check_finite=False),y+h*init,check_finite=False)
                emitted=ac*escape*y
                removed=(weights@loss)*y
                event_rates=np.einsum('k,rki,i->r',weights,flux,y)
            assert y.min()>=-1e-8 and np.isfinite(y).all()
            out=np.bincount(q+1,weights=emitted,minlength=3)
            streams.append([time[j]+h*i,h,*out])
            ext=h*removed.sum()
            external+=h*np.array([prod@n[seeds]-removed@n-out.sum()*(boundary+1),
                                   prod@m[seeds]-removed@m-emitted@m])
            budget[:4]+=h*event_rates
            budget[4:7]+=[h*prod.sum(),h*out.sum(),ext]
            composition+=h*np.array([y.sum(),y@m,y@n,y@(m>=n)])
        close=y.sum()-stores-budget[4]+budget[5]+budget[6]
        species_res=np.array([y@n,y@m])-initial_species-external-(budget[[0,2]]-budget[[1,3]])
        logs.append([time[j+1],stage[j+1],*budget[:7],close,*composition,*species_res])
    log=pd.DataFrame(logs,columns=['end_s','stage','acid_in','acid_out','NH3_in','NH3_out','source','exit','loss','number_error','exposure','ammonia_exposure','sulfate_exposure','paired_exposure','sulfate_error','ammonia_error'])
    relative=abs(log.number_error.sum())/max(1,log.source.sum())
    assert relative<1e-6,relative
    scale=max(1.,log.source.sum())
    assert abs(log.sulfate_error).sum()/scale<1e-6
    if not cfg.get('charge_projection'):assert abs(log.ammonia_error).sum()/scale<1e-6
    return np.array(streams),log
