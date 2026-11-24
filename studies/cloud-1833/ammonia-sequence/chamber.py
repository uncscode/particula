"""Independent cluster and aerosol populations under frozen CLOUD forcing."""
import argparse
import ctypes as ct
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from physics import (PACKET,T,P,amu,capture,common_input,d,e,geometry,par,salt,
                     segments,solid_mass,stage,t,y,missing)

ix=segments['test']
hours=t[ix]
seconds=(hours-hours[0])*3600
st=stage[ix]
gas=np.load(PACKET/'inverse-model/final-gas.npz')['test_288_1_1_0_80'][:,0]
archive_flux=np.load(PACKET/'inverse-model/final-gas.npz')['test_288_1_1_0_80'][:,1]
fit=pd.read_csv(PACKET/'inverse-model/frozen-final.csv').query('k==1 and entry==1').iloc[0]
blend=fit.collision_fraction
ws=np.exp(fit.log_particle_wall)
start11=np.flatnonzero(st==11)[0]
ref=gas[start11-5:start11].mean()
freeze=gas.copy()
freeze[st==11]=ref

obs=y[ix]@np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))
obs_first=[hours[np.flatnonzero(obs>=v)[0]] for v in (1000,5000)]

def reaction(acid,charged):
    cube=(acid/6e7)**3
    if not charged:return 16.59*np.exp(fit.log_neutral_birth)*cube
    rate=132.2*np.exp(fit.log_ion_birth)*cube
    return rate*7/(7+rate)

def wall(diam):
    diff=par.particles.get_diffusion_coefficient_via_system_state(diam*.5e-9,T,P)
    return .002*np.sqrt(diff/(8e-6*(T/293.15)**1.75))

def rate(diam):
    hard,attractive=capture(diam)
    return 1.1*(hard+blend*(attractive-hard))

HERE=Path(__file__).resolve().parent.parent/'charge-evaporation'
ions_ref=pd.read_csv(HERE/'ion-reference.csv')
ions_frame=pd.merge_asof(pd.DataFrame({'archive_h':hours}),ions_ref.sort_values('archive_h'),
                         on='archive_h',tolerance=.001,direction='nearest')
assert np.array_equal(ions_frame.stage,st)
ions=ions_frame[['primary_negative_cm3','primary_positive_cm3']].to_numpy()
field=np.isin(st,[9,11,14])
air=np.ascontiguousarray(np.column_stack([ions,field]).astype(float))
assert np.isfinite(air).all() and air.min()>=0
thermo=pd.read_csv(HERE/'evaporation-reference.csv').set_index(['sulfate_equivalents','ammonia_equivalents','charge_e'])
ledger=[];signed=[];signed_stage=[];gas_return=[]
def charge_rates(diam):
    # Tabulated primary-ion rates from CLOUD SR30-33,39-44,53-54,60-61.
    knots=np.log([2.,5.,15.])
    attach=np.exp(np.interp(np.log(diam),knots,np.log([2e-9,3e-8,1e-7])))
    nt_pos=np.interp(np.log(diam),knots,[1.45e-6,1.75e-6,1.75e-6])
    nt_neg=np.interp(np.log(diam),knots,[1.75e-6,1.45e-6,1.45e-6])
    # Electrical mobility is proportional to the size-dependent diffusion
    # coefficient. Normalize to 1/s at a 1 nm primary ion.
    clear=(wall(diam)/wall(np.array([1.]))[0])**2
    return np.column_stack([attach,nt_pos,nt_neg,clear])

def population(events,z,volume,step,grid,discrete,seed,charge):
    born,charged,period,group,_=events
    ok=z[:,1]==1
    accepted=z[ok];g=group[ok]
    order=np.lexsort((accepted[:,0],g))
    offsets=np.r_[0,np.cumsum(np.bincount(g,minlength=boxes))].astype('i4')
    # Conserve dry sulfate at the hydration-law transition. Incoming particles
    # can be fractionally larger than 2.9 nm after re-equilibration.
    diam=np.geomspace(2.87,96,grid)
    mass=solid_mass(diam,'aiomfac')/(salt*amu)
    physical=1350*np.pi/6*(diam*1e-9)**3
    kernel=par.dynamics.get_brownian_kernel_via_system_state(diam*.5e-9,physical,T,P)*1e6
    g0=geometry(288,'aiomfac',1.)
    initial=y[ix[0]].copy();initial[3]=np.interp(np.log(d[3]),np.log(d[[2,4]]),initial[[2,4]])
    census=g0['mi']@initial
    initial_data=np.r_[census,solid_mass(g0['D'],'aiomfac')/(salt*amu)]
    inputs=[seconds,gas,mass,diam,rate(diam),wall(diam),kernel,initial_data,offsets,accepted[order,0],accepted[order,2],10**e]
    inputs=[np.ascontiguousarray(v) for v in inputs]
    result=np.zeros((boxes,len(hours),len(d)+1));audit=np.zeros((boxes,3))
    lib=ct.CDLL(str(LIB/'particles.so'))
    lib.population.argtypes=[ct.c_int]*5+[ct.c_double]*3+[ct.c_void_p]*12+[ct.c_int,ct.c_uint,ct.c_void_p,ct.c_void_p]
    sign_out=np.zeros((boxes,len(hours),4,2));charge_audit=np.zeros((boxes,4))
    lookup=np.ascontiguousarray(charge_rates(diam));sign_in=np.ascontiguousarray(accepted[order,3])
    lib.population.argtypes += [ct.c_int]+[ct.c_void_p]*5+[ct.c_double]
    # Upper AIS bound set below using the authors' measurement grid.
    rc=lib.population(boxes,len(hours),grid,len(census),discrete,volume,step,ws,
       *[v.ctypes.data for v in inputs],len(d),seed,result.ctypes.data,audit.ctypes.data,charge,lookup.ctypes.data,air.ctypes.data,sign_in.ctypes.data,sign_out.ctypes.data,charge_audit.ctypes.data,AIS_TOP)
    assert rc==0 and audit[:,0].max()<1e-9 and (audit[:,1]==0).all()
    assert np.isfinite(result).all() and np.all(charge_audit[:,0]==0)
    return result/volume,audit,sign_out,charge_audit


AIS_TOP=40.
