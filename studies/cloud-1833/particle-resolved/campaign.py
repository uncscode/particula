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

cli=argparse.ArgumentParser()
cli.add_argument('--out',type=Path,required=True)
cli.add_argument('--lib',type=Path,required=True)
cli.add_argument('--boxes',type=int,default=64)
cli.add_argument('--pilot',action='store_true')
args=cli.parse_args()
args.out.mkdir(exist_ok=True,parents=True)
tic=time.monotonic()
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
boxes=args.boxes
obs=y[ix]@np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))
obs_first=[hours[np.flatnonzero(obs>=v)[0]] for v in (1000,5000)]

def reaction(acid,charged):
    cube=(acid/6e7)**3
    if not charged:return 16.59*np.exp(fit.log_neutral_birth)*cube
    rate=132.2*np.exp(fit.log_ion_birth)*cube
    return rate*7/(7+rate)

def births(birth_acid,volume,seed):
    rng=np.random.default_rng(seed)
    times=[];flags=[];intervals=[];groups=[]
    for j,delta in enumerate(np.diff(seconds)):
        for half in (0,1):
            for charged in (False,True):
                if charged and st[j+half] not in (10,12,15,17):continue
                limit=reaction(max(birth_acid[j:j+2]),charged)
                count=rng.poisson(volume*boxes*limit*delta/2)
                position=(half+rng.random(count))/2
                acid=birth_acid[j]+position*(birth_acid[j+1]-birth_acid[j])
                keep=rng.random(count)*limit<reaction(acid,charged)
                frac=position[keep];count=len(frac)
                times.append(seconds[j]+frac*delta);flags.append(np.full(count,charged))
                intervals.append(np.full(count,j,dtype='i4'));groups.append(rng.integers(boxes,size=count,dtype='i4'))
    born,charged,period,group=map(np.concatenate,(times,flags,intervals,groups))
    seeds=rng.integers(1,2**63-1,len(born),dtype='uint64')
    return born,charged,period,group,seeds

def wall(diam):
    diff=par.particles.get_diffusion_coefficient_via_system_state(diam*.5e-9,T,P)
    return .002*np.sqrt(diff/(8e-6*(T/293.15)**1.75))

def rate(diam):
    hard,attractive=capture(diam)
    return 1.1*(hard+blend*(attractive-hard))

def cluster_track(events,acid,discrete):
    born,charged,period,group,seeds=events
    masscoef=1350*np.pi/6*1e-27/((salt+18.01528)*amu)
    lo,hi=masscoef*np.array([.8,2.9])**3
    du=.05
    mass=lo+du*np.arange(int(np.ceil((hi-lo)/du))+3)
    diam=(mass/masscoef)**(1/3)
    host=[seconds,acid,rate(diam),ws*wall(diam)+.0001754037,born,period,seeds]
    device=[wp.array(np.ascontiguousarray(x),device='cuda:0') for x in host]
    output=wp.empty((len(born),3),dtype=wp.float64,device='cuda:0')
    lib=ct.CDLL(str(args.lib/'cluster.so'))
    lib.clusters.argtypes=[ct.c_int]*3+[ct.c_double]*3+[ct.POINTER(ct.c_void_p)]
    ptrs=(ct.c_void_p*8)(*[x.ptr for x in device+[output]])
    wp.synchronize()
    assert lib.clusters(len(born),len(seconds),discrete,du,lo,hi,ptrs)==0
    z=output.numpy()
    assert np.isfinite(z).all() and (z[:,0]>=born-1e-6).all()
    assert np.isin(z[:,1],[-1,0,1]).all()
    assert np.all(z[z[:,1]==1,2]>=hi-1e-10)
    return z

def population(events,z,volume,step,grid,discrete,seed):
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
    lib=ct.CDLL(str(args.lib/'particles.so'))
    lib.population.argtypes=[ct.c_int]*5+[ct.c_double]*3+[ct.c_void_p]*12+[ct.c_int,ct.c_uint,ct.c_void_p,ct.c_void_p]
    rc=lib.population(boxes,len(hours),grid,len(census),discrete,volume,step,ws,
       *[v.ctypes.data for v in inputs],len(d),seed,result.ctypes.data,audit.ctypes.data)
    assert rc==0 and audit[:,0].max()<1e-9 and (audit[:,1]==0).all()
    assert audit[:,2].sum()>0 and np.isfinite(result).all()
    return result/volume,audit

summary=[];cohorts=[];series={};cohortflux=[];boot_records=[];spectral=[]
def keep(name,events,z,volume,step,grid,discrete):
    born,charged,period,group,_=events
    live=z[:,1]==1
    stage_birth=st[period+(born-seconds[period]>=np.diff(seconds)[period]/2)]
    arrivals=z[live,0]
    stage_index=np.clip(np.searchsorted((seconds[1:]+seconds[:-1])/2,arrivals),0,len(st)-1)
    stage_arrive=st[stage_index]
    for s in range(9,19):
        origins=stage_birth==s
        for pathway,subset in [('all',origins),('neutral',origins&~charged),('charged',origins&charged)]:
            if not subset.any():continue
            ok=subset&live
            lag=(z[ok,0]-born[ok])/60
            cohorts.append(dict(case=name,stage=s,pathway=pathway,births_per_cm3=subset.sum()/volume/boxes,
                fraction_reaching=ok.sum()/subset.sum(),fraction_lost=(subset&(z[:,1]==-1)).sum()/subset.sum(),
                fraction_still_in_chamber=(subset&(z[:,1]==0)).sum()/subset.sum(),
                median_min=np.median(lag) if len(lag) else np.nan,
                p10_min=np.quantile(lag,.1) if len(lag) else np.nan,p90_min=np.quantile(lag,.9) if len(lag) else np.nan))
        cohortflux.append(dict(case=name,stage=s,entering_cm3=sum(stage_arrive==s)/volume/boxes,
            fraction_from_prior_stages=np.mean(stage_birth[live][stage_arrive==s]<s) if np.any(stage_arrive==s) else np.nan))
    result,audit=population(events,z,volume,step,grid,discrete,183304)
    curves=result[:,:,-1];mean=curves.mean(0)
    binned=result[:,:,:-1].mean(0)/np.diff(e)
    def times(a):
        o=[]
        for v in [1000,5000]:
            hits=np.flatnonzero(a>=v);o.append(hours[hits[0]] if len(hits) else np.nan)
        return np.array(o)
    delay=60*(times(mean)-obs_first)
    rng=np.random.default_rng(183305)
    bs=np.array([times(curves[rng.integers(0,boxes,boxes)].mean(0)) for _ in range(1500)])
    rise=(bs[:,1]-bs[:,0])*60
    incl=st==12;mode=np.median(d[4:][np.argmax(binned[incl,4:],axis=1)])
    widths=np.diff(e)
    for sid in range(9,19):
        mask=(st==sid)
        scored=mask & ~missing[ix]
        den=binned[scored];refden=y[ix[scored]]
        absolute=np.abs(np.log10(den[:,4:]+1)-np.log10(refden[:,4:]+1))
        factor=10**np.mean(absolute@widths[4:]/widths[4:].sum())
        spectral.append(dict(case=name,stage=sid,spectral_factor_4_to_64nm=factor,
            mean_4nm_bin_cm3=den[:,4].mean()*widths[4],observed_4nm_bin_cm3=refden[:,4].mean()*widths[4],
            mean_5p9nm_bin_cm3=den[:,5].mean()*widths[5],observed_5p9nm_bin_cm3=refden[:,5].mean()*widths[5],
            initial_tail_cm3=mean[mask][0],last_tail_cm3=mean[mask][-1],observed_last_tail_cm3=obs[mask][-1]))
    flux=np.histogram(arrivals,bins=seconds)[0]/np.diff(seconds)/boxes/volume
    series[name]=mean
    series[name+'_entry']=np.r_[np.nan,flux]
    summary.append(dict(case=name,boxes=boxes,volume_cm3=volume,step_s=step,lookup_sizes=grid,births=len(born),
        admitted=live.sum(),rise_min=np.diff(times(mean))[0]*60,first_error_min=delay[0],second_error_min=delay[1],
        rise_mc_low=np.nanquantile(rise,.025),rise_mc_high=np.nanquantile(rise,.975),
        mode12_nm=mode,stage11_influx_cm3_s=flux[st[1:]==11].mean(),
        stage14_influx_cm3_s=flux[st[1:]==14].mean(),max_sulfate_budget_error=audit[:,0].max(),mergers=int(audit[:,2].sum())))
    for q,time_bs in zip(['first','second'],bs.T):
        boot_records.append(dict(case=name,threshold=q,p025_h=np.nanquantile(time_bs,.025),p975_h=np.nanquantile(time_bs,.975),fraction_crossed=np.isfinite(time_bs).mean()))
    print(name,round(summary[-1]['rise_min'],1),round(summary[-1]['stage11_influx_cm3_s'],2),flush=True)

def run():
    volume=.04;step=8;grid=384
    assert abs(d[4]-4.)<.01 and abs(d[5]-5.9)<.01
    base=births(gas,volume,183300)
    jumps=cluster_track(base,gas,1)
    keep('discrete',base,jumps,volume,step,grid,1)
    if not args.pilot:
        keep('continuous',base,cluster_track(base,gas,0),volume,step,grid,0)
        keep('discrete clusters, continuous aerosol',base,jumps,volume,step,grid,0)
        keep('fixed cluster growth in stage 11',base,cluster_track(base,freeze,1),volume,step,grid,1)
        other=births(freeze,volume,183301)
        keep('fixed birth acid in stage 11',other,cluster_track(other,gas,1),volume,step,grid,1)
        keep('fixed birth and growth acid in stage 11',other,cluster_track(other,freeze,1),volume,step,grid,1)
        halfway=(gas+freeze)/2
        half_events=births(halfway,volume,183306)
        keep('half birth acid response in stage 11',half_events,cluster_track(half_events,gas,1),volume,step,grid,1)
        keep('half growth acid response in stage 11',base,cluster_track(base,halfway,1),volume,step,grid,1)
        keep('finer timestep and table',base,jumps,volume,step/2,2*grid,1)
        large=births(gas,volume*2,183303)
        keep('double box volume',large,cluster_track(large,gas,1),volume*2,step,grid,1)
    pd.DataFrame(summary).to_csv(args.out/'cases.csv',index=False)
    pd.DataFrame(cohorts).to_csv(args.out/'cohorts.csv',index=False)
    pd.DataFrame(spectral).to_csv(args.out/'stage-spectra.csv',index=False)
    pd.DataFrame(cohortflux).to_csv(args.out/'arrivals.csv',index=False)
    pd.DataFrame(boot_records).to_csv(args.out/'bootstrap.csv',index=False)
    pd.DataFrame(dict(archive_h=hours,observed_tail=obs,sectional_inlet=archive_flux,**series)).to_csv(args.out/'timeseries.csv',index=False)
    (args.out/'run.json').write_text(json.dumps(dict(seconds=time.monotonic()-tic,device=wp.get_device('cuda:0').name,
       downstream='C++ OpenMP CPU',boxes_per_case=boxes,stage11_prior_acid=ref,
       source='test_288_1_1_0_80',cluster_law='one associated water per neutralized sulfate',seeds=[183300,183301,183303,183304,183305,183306]),indent=2))
run()
