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
cli.add_argument('--capture-checks',action='store_true')
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

HERE=Path(__file__).resolve().parent
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

def cluster_track(events,charge,law,absorb,birth_size=.8):
    born,charged,period,group,seeds=events
    masscoef=1350*np.pi/6*1e-27/((salt+18.01528)*amu)
    lo,hi=masscoef*np.array([birth_size,2.9])**3
    mass=lo+np.arange(int(np.ceil(hi-lo))+1)
    diam=(mass/masscoef)**(1/3)
    coeff=np.column_stack([rate(diam),ws*wall(diam)+.0001754037,charge_rates(diam),rate(diam),rate(diam)])
    evap=np.zeros((len(mass),3))
    if law!='off' and birth_size==.8:
        for n in (2,3,4):
            for q in (-1,0,1):
                if q<0:
                    ammonia=n-1
                elif law in ('paired','matched_paired'):
                    ammonia=n+int(q>0)
                else:
                    ammonia=n-1+int(q>0)
                gamma=thermo.loc[(n,ammonia,q),'single_H2SO4_loss_s-1']
                if law in ('chamber_fit','matched_chamber_fit') and q==0:
                    free_energy=thermo.loc[(n,ammonia,q),'delta_G_add_acid_kcal_mol']
                    fitted={2:-15.,3:-16.8,4:-15.7}[n]
                    gamma*=np.exp((fitted-free_energy)/(.00198720425864*298.15))
                if law in ('matched','capture_only','matched_paired','matched_chamber_fit','signed_capture_only'):
                    column={-1:6,0:0,1:7}[q]
                    if q or law!='signed_capture_only':
                        coeff[n-2,column]=thermo.loc[(n,ammonia,q),'acid_capture_beta_cm3_s']
                    if law in ('capture_only','signed_capture_only'): gamma=0.
                if law=='partial_hydrates':
                    from evaporation_reference import hydrate_factor
                    factor=hydrate_factor(n,ammonia,q,1800.55)
                    if factor is not None: gamma*=factor
                evap[n-1,q+1]=gamma
    # Fixed seeds and shared charged birth histories in every scenario.
    birth_sample=period+(born>=(seconds[period]+seconds[period+1])*.5)
    neg,pos=air[birth_sample,:2].T
    p_pos=np.divide(.1*pos,neg+.1*pos,out=np.full(len(born),1/11),where=neg+.1*pos>0)
    birth_q=charged*np.where(np.random.default_rng(183330).random(len(born))<p_pos,1,-1)
    host=[seconds,gas,coeff,air,evap,born,period,birth_q.astype('i4'),seeds]
    device=[wp.array(np.ascontiguousarray(x),device='cuda:0') for x in host]
    output=wp.empty((len(born),8),dtype=wp.float64,device='cuda:0')
    counts=wp.zeros((len(seconds),2),dtype=wp.int32,device='cuda:0')
    lib=ct.CDLL(str(args.lib/'cluster.so'))
    lib.clusters.argtypes=[ct.c_int]*4+[ct.c_double]*2+[ct.POINTER(ct.c_void_p)]
    ptrs=(ct.c_void_p*11)(*[x.ptr for x in device+[output,counts]])
    wp.synchronize()
    assert lib.clusters(len(born),len(seconds),charge,absorb,lo,hi,ptrs)==0
    z=output.numpy()
    assert np.isfinite(z).all() and (z[:,0]>=born-1e-6).all()
    assert np.all(z[z[:,1]==1,2]>=hi)
    np.testing.assert_allclose(z[:,2],lo+z[:,4]-z[:,5],atol=1e-10)
    assert np.isin(z[:,1],[-3,-2,-1,0,1]).all() and np.isin(z[:,3],[-1,0,1]).all()
    return z,counts.numpy()
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
    lib=ct.CDLL(str(args.lib/'particles.so'))
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

summary=[];cohorts=[];series={};cohortflux=[];boot_records=[];spectral=[]
def keep(name,events,outputs,volume,step,grid,charge):
    z,subcounts=outputs
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
                fraction_reaching=ok.sum()/subset.sum(),fraction_lost=(subset&(z[:,1]<0)).sum()/subset.sum(),
                fraction_still_in_chamber=(subset&(z[:,1]==0)).sum()/subset.sum(),
                median_min=np.median(lag) if len(lag) else np.nan,
                p10_min=np.quantile(lag,.1) if len(lag) else np.nan,p90_min=np.quantile(lag,.9) if len(lag) else np.nan))
        cohortflux.append(dict(case=name,stage=s,entering_cm3=sum(stage_arrive==s)/volume/boxes,
            fraction_from_prior_stages=np.mean(stage_birth[live][stage_arrive==s]<s) if np.any(stage_arrive==s) else np.nan))
    result,audit,sign_out,qa=population(events,z,volume,step,grid,1,183304,charge)
    ledger.append(dict(case=name,births=len(born),admitted=int(live.sum()),in_transit=int(sum(z[:,1]==0)),
        cluster_background_loss=int(sum(z[:,1]==-1)),cluster_field_loss=int(sum(z[:,1]==-2)),cluster_dissociation=int(sum(z[:,1]==-3)),
        cluster_uptakes=int(z[:,4].sum()),acid_evaporations=int(z[:,5].sum()),cluster_chargings=int(z[:,6].sum()),
        cluster_neutralizations=int(z[:,7].sum()),downstream_field_loss=int(qa[:,1].sum()),downstream_chargings=int(qa[:,2].sum()),
        downstream_neutralizations=int(qa[:,3].sum()),max_downstream_net_charge_residual=int(abs(qa[:,0]).max())))
    bands=sign_out.sum(0)
    bands[:,0,:]+=subcounts
    bands/=boxes*volume
    pred=bands.sum(1)
    ranges=['2p2_to_2p9','2p9_to_5','5_to_9p7','9p7_to_40']
    for i,sid in enumerate(st):
        signed.append(dict(case=name,archive_h=hours[i],stage=sid,negative_cm3=pred[i,0],positive_cm3=pred[i,1],
            **{f"{pol}_{span}_cm3":bands[i,b,k] for b,span in enumerate(ranges) for k,pol in enumerate(["negative","positive"])}))
    for sid in range(9,19):
        mask=st==sid
        for pol,column in enumerate(['ais_negative_ge2p2_cm3','ais_positive_ge2p2_cm3']):
            observations=ions_frame[column].to_numpy();valid=mask & np.isfinite(observations)
            if not valid.any():continue
            signed_stage.append(dict(case=name,stage=sid,polarity=['negative','positive'][pol],samples=valid.sum(),
                modeled_mean_cm3=pred[valid,pol].mean(),observed_mean_cm3=observations[valid].mean(),
                mae_cm3=abs(pred[valid,pol]-observations[valid]).mean()))

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
        rise_mc_low=safeq(rise,.025),rise_mc_high=safeq(rise,.975),
        mode12_nm=mode,stage11_influx_cm3_s=flux[st[1:]==11].mean(),
        stage14_influx_cm3_s=flux[st[1:]==14].mean(),max_sulfate_budget_error=audit[:,0].max(),mergers=int(audit[:,2].sum())))
    for q,time_bs in zip(['first','second'],bs.T):
        boot_records.append(dict(case=name,threshold=q,p025_h=safeq(time_bs,.025),p975_h=safeq(time_bs,.975),fraction_crossed=np.isfinite(time_bs).mean()))
    print(name,round(summary[-1]['rise_min'],1),round(summary[-1]['stage11_influx_cm3_s'],2),flush=True)

def safeq(arr,q):
    a=arr[np.isfinite(arr)]
    return np.quantile(a,q) if len(a) else np.nan

def run():
    volume=.04
    events=births(gas,volume,183300)
    cases=[('reference',False,'off',False),('charge',True,'off',False),
           ('evaporation',False,'acid_rich',False),('both',True,'acid_rich',False)]
    if not args.pilot:
        cases.extend([('evaporation_paired',False,'paired',False),('both_paired',True,'paired',False),
                      ('both_dimer_dissociates',True,'acid_rich',True),('both_partial_hydrates',True,'partial_hydrates',False),
                      ('evaporation_chamber_fit',False,'chamber_fit',False),('both_chamber_fit',True,'chamber_fit',False),
                      ('charge_molecular_capture',True,'capture_only',False),('both_matched_capture',True,'matched',False)])
    capture_checks=[('molecular_capture_no_charge',False,'capture_only',False),
                    ('charged_capture_change_only',True,'signed_capture_only',False),
                    ('both_paired_matched_capture',True,'matched_paired',False),
                    ('both_chamber_fit_matched_capture',True,'matched_chamber_fit',False)]
    if args.capture_checks:
        cases=[('charge_molecular_capture',True,'capture_only',False)]+capture_checks
    elif not args.pilot:
        cases+=capture_checks
    for name,charge,law,absorb in cases:
        z=cluster_track(events,charge,law,absorb)
        keep(name,events,z,volume,8,384,charge)
        if name in ['charge','both','both_paired','both_chamber_fit','charge_molecular_capture','both_paired_matched_capture','both_chamber_fit_matched_capture'] and not args.pilot:
            keep(name+'_refined',events,z,volume,4,768,charge)
    if not args.pilot:
        large=births(gas,volume*2,183303)
        for name,law in [('molecular_capture_larger_box','capture_only'),('both_chamber_fit_matched_larger_box','matched_chamber_fit')]:
            z=cluster_track(large,True,law,False)
            keep(name,large,z,volume*2,8,384,True)
    if not (args.pilot or args.capture_checks):
        # Separate source interpretation: form embryos at the stated J1.7 diameter.
        # Every available 2-4-sulfate reverse channel then lies below this source.
        for name,active in [('source_1p7',False),('source_1p7_charge',True)]:
            z=cluster_track(events,active,'off',False,1.7)
            keep(name,events,z,volume,8,384,active)
        nominal=air.copy()
        air[:,:2]=ions_frame[['primary_negative_offset150_cm3','primary_positive_offset150_cm3']]
        air[field,:2]=7.
        z=cluster_track(events,True,'off',False)
        keep('charge_ion_forcing_sensitivity',events,z,volume,8,384,True)
        air[:]=nominal
        large=births(gas,volume*2,183303)
        for name,law in [('both_larger_box','acid_rich'),('both_paired_larger_box','paired')]:
            z=cluster_track(large,True,law,False)
            keep(name,large,z,volume*2,8,384,True)
    pd.DataFrame(summary).to_csv(args.out/'cases.csv',index=False)
    pd.DataFrame(cohorts).to_csv(args.out/'cohorts.csv',index=False)
    pd.DataFrame(spectral).to_csv(args.out/'stage-spectra.csv',index=False)
    pd.DataFrame(cohortflux).to_csv(args.out/'arrivals.csv',index=False)
    pd.DataFrame(boot_records).to_csv(args.out/'bootstrap.csv',index=False)
    pd.DataFrame(signed).to_csv(args.out/'charge-timeseries.csv',index=False)
    pd.DataFrame(signed_stage).to_csv(args.out/'charge-by-stage.csv',index=False)
    pd.DataFrame(ledger).to_csv(args.out/'events.csv',index=False)
    pd.DataFrame(dict(archive_h=hours,observed_tail=obs,**series)).to_csv(args.out/'timeseries.csv',index=False)
    (args.out/'run.json').write_text(json.dumps(dict(seconds=time.monotonic()-tic,device=wp.get_device('cuda:0').name,
        downstream='C++ OpenMP CPU',boxes_per_case=boxes,number_of_cases=len(summary),
        source='test_288_1_1_0_80',lower_cut_nm=2.2,ais_upper_nm=AIS_TOP,
        seeds=[183300,183303,183304,183305,183330]),indent=2))
AIS_TOP=40.
run()
