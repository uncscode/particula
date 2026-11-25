# %%
"""Pulse replacement in isothermal boxes, Particula expectation and CUDA event ensembles."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
import particula as par
from particula import BoxProcess, FieldSpec
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from study import read_inputs,geometry,train,weibull,survival_law

FIELDS=('lund.original','lund.fresh','lund.ice','lund.exchanged')


class IcePulse(BoxProcess):
    def __init__(self,survival,fractions,reset):
        self.survival=survival
        self.fractions=fractions
        self.reset=reset
        super().__init__(name='freeze',extension_id='study.lund.icepulse',
                         reads=FIELDS+('environment.temperature',),writes=FIELDS)

    def advance(self,aerosol,*,time,time_step,context):
        box=int(context.box_id)
        start=int(round(time/3600)); end=int(round((time+time_step)/3600))
        s=self.survival[box]
        original,fresh,ice,exchanged=(aerosol.fields[key] for key in FIELDS)
        if start==2:
            removal=original[0]*self.fractions[box]
            original[0]-=removal
            fresh[0]+=removal
            exchanged[0]+=removal
        offset=2 if self.reset[box] and start>=2 else 0
        old=original[0]+fresh[0]
        original[0]*=s[end]/s[start]
        fresh[0]*=s[end-offset]/s[start-offset]
        ice[0]+=old-(original[0]+fresh[0])


def state_for(temps):
    n=len(temps)
    state=par.Aerosol(
        particles=ParticleData(masses=np.full((n,1,1),1e-18),concentration=np.ones((n,1)),
                               charge=np.zeros((n,1)),density=np.array([1000.]),
                               volume=np.full(n,.001),distribution_type='particle_resolved'),
        gas=GasData(name=['inert'],molar_mass=np.array([.029]),concentration=np.zeros((n,1)),
                    partitioning=np.array([False])),
        environment=EnvironmentData(temperature=temps+273.15,pressure=np.full(n,101325.),
                                    saturation_ratio=np.ones((n,1))))
    for key in FIELDS:
        # Scenario carriers and ice are counted in explicit extension fields.
        state.fields.add(FieldSpec(key,('box',),'L^-1',initial_value=float(key==FIELDS[0]),minimum=0))
    return state


def inputs(pkt,pilot):
    raw,_,_=read_inputs(pkt,False); raw=geometry(raw)
    rows=[]; curves=[]; interventions=[]
    hours=np.arange(11)
    for name,sample in raw.groupby('sample'):
        if pilot and name!='Marine':
            continue
        laws=[('persistent',1.,0.),('aging',.6,0.),('redrawn',1.,.6)]
        fits=[(model,b,k,train(sample,power=b,gamma=k)) for model,b,k in laws]
        for plate,night in sample[(sample.phase=='night')&(sample.h>0)].groupby('plate'):
            if pilot and plate!=1:
                continue
            i=((plate-1)*100+night.id-1).to_numpy()
            survival_curves=[]
            for model,b,k,(ref,rates,prior,_,posterior) in fits:
                w=posterior[i].copy();w[night['shift']>80]=prior
                initial=w.mean(axis=0)
                by_rate=survival_law(hours,rates,b,k)
                surv=by_rate @ initial
                survival_curves.append((model,surv))
                # Retain the *types* of the two-hour survivors. A complete pulse
                # reset discards within-cycle elapsed time or redraws its multiplier.
                pulse=float((by_rate[2]*(1-by_rate[8])) @ initial)
                uninterrupted=float(surv[2]-surv[10])
                interventions.append(dict(sample=name,plate=plate,model=model,paused=uninterrupted,
                                          restarted=pulse,increment=pulse-uninterrupted))
            survival_curves.append(('population',1-weibull(hours,fits[0][3][0])))
            for model,s in survival_curves:
                for reset in [False,True]:
                    for fraction in [0,.25,.5,1]:
                        rows.append(dict(sample=name,plate=plate,model=model,reset=reset,fraction=fraction,
                                         target_c=sample.target.iloc[0],s2=s[2],s8=s[8],s10=s[10]))
                        curves.append(s)
    return pd.DataFrame(rows),np.asarray(curves),pd.DataFrame(interventions)


def deterministic(cases,curves,dt):
    sim=IcePulse(curves,cases.fraction.to_numpy(),cases.reset.to_numpy())
    state=state_for(cases.target_c.to_numpy())
    for t in range(0,10,dt):
        sim.execute(state,dt*3600,time=t*3600)
    old,new,ice,swapped=[state.fields[key] for key in FIELDS]
    np.testing.assert_allclose(old+new+ice,1,rtol=0,atol=3e-14)
    np.testing.assert_allclose(swapped,cases.fraction*cases.s2,rtol=0,atol=3e-14)
    return ice-(1-cases.s2.to_numpy())


@wp.kernel
def simulate(count:int,reps:int,seed:int,params:wp.array2d(dtype=wp.float64),
             plate_id:wp.array(dtype=wp.int32), result:wp.array2d(dtype=wp.int32)):
    case,rep=wp.tid()
    rng=wp.rand_init(seed,plate_id[case]*reps+rep)
    ice=int(0)
    for _ in range(count):
        u0=wp.randf(rng)
        exchange=wp.randf(rng)
        u1=wp.randf(rng)
        if wp.float64(u0)<params[case,0]:
            p=params[case,1]
            if wp.float64(exchange)<params[case,3]:
                p=params[case,2]
            if wp.float64(u1)<p:
                ice+=1
    result[case,rep]=ice


def campaign(pkt,out,pilot):
    cases,curves,pulse=inputs(pkt,pilot)
    exact=deterministic(cases,curves,2)
    fine=deterministic(cases,curves,1)
    oldp=1-cases.s10/cases.s2
    freshp=np.where(cases.reset,1-cases.s8,oldp)
    theory=cases.s2*((1-cases.fraction)*oldp+cases.fraction*freshp)
    np.testing.assert_allclose(exact,theory,atol=3e-14)
    np.testing.assert_allclose(fine,exact,atol=3e-14)
    cases['late_ice_per_l']=exact
    cases['refined_ice_per_l']=fine
    wp.init()
    device=wp.get_device("cuda:0");assert device.is_cuda
    params=wp.array(np.column_stack([cases.s2,oldp,freshp,cases.fraction]),dtype=wp.float64,device=device)
    plate_id=cases.groupby(['sample','plate'],sort=True).ngroup().to_numpy(dtype=np.int32)
    plates=wp.array(plate_id,dtype=wp.int32,device=device)
    rep=16 if pilot else 96
    counts=[128] if pilot else [512,2048]
    all_results=[]; seconds=[]
    for n in counts:
        seed=251125+n
        result=wp.zeros((len(cases),rep),dtype=wp.int32,device=device)
        wp.synchronize_device(device); start=time.monotonic()
        wp.launch(simulate,dim=(len(cases),rep),inputs=[n,rep,seed,params,plates,result],device=device)
        wp.synchronize_device(device);seconds.append(time.monotonic()-start)
        values=result.numpy()
        assert np.all((values>=0)&(values<=n))
        sim=values/n; cases[f'mc_mean_{n}']=sim.mean(axis=1)
        se=sim.std(axis=1,ddof=1)/np.sqrt(rep)
        cases[f'mc_se_{n}']=se
        if not pilot:
            assert np.all(abs(sim.mean(axis=1)-exact)<5*se+1/(rep*n))
        all_results.append(values)
    tag='box_pilot' if pilot else 'boxes'
    cases.to_csv(out/(tag+'.csv'),index=False)
    pulse.to_csv(out/(tag+'_retained_pulse.csv'),index=False)
    np.savez_compressed(out/(tag+'_counts.npz'),**{f'n{n}':v for n,v in zip(counts,all_results)})
    report=dict(device=device.name,warp=wp.__version__,particula=par.__version__,
                source=str(Path(par.__file__).resolve()),cases=len(cases),reps=rep,particles=counts,
                seeds=[251125+n for n in counts],ensemble_boxes=len(cases)*rep*len(counts),
                simulated_carriers=len(cases)*rep*sum(counts),kernel_wall_seconds=seconds,
                largest_dt_difference=float(np.max(abs(exact-fine))))
    (out/(tag+'_run.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(report)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--pilot',action='store_true')
    a=parser.parse_args();campaign(a.input,a.output,a.pilot)
