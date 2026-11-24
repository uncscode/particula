"""A matched inlet shared by each downstream CUDA chamber in a batch."""
import argparse,ctypes as ct,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import warp as wp
from domain import geometry
from boundary import reconstruct,wall_scale,blend,ix,st,hours,seconds,obs,acid,gap
from physics import d,e,PACKET,overlap

parser=argparse.ArgumentParser()
parser.add_argument('out',type=Path);parser.add_argument('lib',type=Path);parser.add_argument('--pilot',action='store_true')
args=parser.parse_args()
start=time.monotonic()
field=reconstruct()
if not args.pilot:
    passage=pd.read_csv(args.out/'passages.csv')
    np.testing.assert_allclose(passage.start_h,hours[:-1])
    field['prior_passages']=np.r_[passage.total_cm3_s,0.]
    early=(st[:-1]<=12)&(st[1:]<=12)
    dt=np.diff(seconds)[early]
    f=field.direct.to_numpy()
    total_direct=(.5*(f[1:]+f[:-1]))[early]@dt
    total_prior=field.prior_passages.to_numpy()[:-1][early]@dt
    field['prior_timing_direct_amount']=field.prior_passages*total_direct/total_prior
    field['direct_timing_prior_amount']=field.direct*total_prior/total_direct
field.to_csv(args.out/'boundary.csv',index=False)
lib=ct.CDLL(str(args.lib))
lib.replay.argtypes=[ct.c_int]*3+[ct.c_double]*2+[ct.POINTER(ct.c_void_p)]
models=['frozen','hard','attractive','weak_band_wall','strong_band_wall','weak_all_wall','prior_wall','weak_tail_wall','prior_wall_hard','prior_wall_attractive']
inlets=['direct_hard','direct','direct_attractive','from4','from5p9','balance','balance_wide','direct_plus_coag','dark_zero','dark_max','prior_passages','prior_timing_direct_amount','direct_timing_prior_amount']
if args.pilot:models=['frozen','hard'];inlets=['direct']
rows=[];stages=[];trajectories=[]
tail=np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))
target=obs@tail
def crosses(a):
    return np.array([hours[np.flatnonzero(a>=v)[0]] if np.any(a>=v) else np.nan for v in [1000,5000]])
observed_times=crosses(target)
def stage_mode(spec):
    # Compare on complete measured bins downstream of both inlet channels.
    return np.median(d[6:][spec[st==12,6:].argmax(1)])
for cells,step in ([(96,12)] if args.pilot else [(192,12),(288,6)]):
    g=geometry(cells,'aiomfac',1.);nt=len(hours)
    walls=g['wall']*wall_scale;edges=10**g['edges'];mid=g['D']
    hard,attractive=g['vel'];base=hard+blend*(attractive-hard)
    initial=g['mi']@obs[0]
    assert np.isclose(initial.sum(),obs[0]@np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(5.))))
    measured_bins=g['mo']/np.diff(e)[:,None]
    # Same partial-bin observation operator as data, plus exact cutoff as a sensitivity.
    frac=np.maximum(0,np.minimum(g['edges'][1:],e[-1])-np.maximum(g['edges'][:-1],np.log10(9.7)))/g['dx']
    for inlet in inlets:
        vel=[];wallset=[]
        for name in models:
            v=base.copy();w=walls.copy()
            if name in ['hard','attractive','prior_wall_hard','prior_wall_attractive']:
                v[edges<9.7]=(hard if name.endswith('hard') else attractive)[edges<9.7]
            if name=='weak_band_wall':w[mid<9.7]*=.5
            if name=='strong_band_wall':w[mid<9.7]*=2
            if name=='weak_all_wall':w*=.5
            if name=='weak_tail_wall':w[mid>=9.7]*=.5
            if name.startswith('prior_wall'):w*=.5/wall_scale
            vel.append(np.r_[v,v.max()]);wallset.append(w)
        count=len(models)
        f=field[inlet].to_numpy();assert np.isfinite(f).all() and f.min()>=0
        host=[hours,np.tile(acid,count),np.ones(nt),initial,np.array(vel),np.array(wallset),
              np.tile([1.1,1.,float(inlet.startswith('prior_')),1.],(count,1)),g['K'],*g['pairs'],g['over'],np.zeros(nt),np.tile(f,count)]
        gpu=[wp.array(np.ascontiguousarray(v),device='cuda:0') for v in host]
        state=wp.empty((count,nt,cells),dtype=wp.float64,device='cuda:0')
        ledger=wp.empty((count,cells,6),dtype=wp.float64,device='cuda:0')
        ptr=(ct.c_void_p*17)(*[v.ptr for v in gpu+[state,ledger]]);wp.synchronize()
        assert lib.replay(count,cells,nt,g['dx'],step,ptr)==0
        x=state.numpy();book=ledger.numpy().sum(1);entered=f[:-1]@np.diff(seconds) if inlet.startswith('prior_') else np.trapezoid(f,seconds)
        assert np.isfinite(x).all() and x.min()>-1e-10
        np.testing.assert_allclose(book[:,0],entered,rtol=1e-10)
        for mi,name in enumerate(models):
            b=book[mi];j=x[mi];np.testing.assert_allclose(j[0],initial)
            residual=abs(j[-1].sum()-initial.sum()-b[0]+b[1:].sum())/max(1,b[0]+initial.sum())
            assert residual<1e-9
            spec=j@measured_bins.T;curve=spec@tail;phys_curve=j@frac
            hits=crosses(curve);delay=(hits-observed_times)*60;dark=st==18
            identifier=dict(cells=cells,step_s=step,inlet=inlet,model=name)
            rows.append(dict(**identifier,first_error_min=delay[0],second_error_min=delay[1],rise_min=np.diff(hits)[0]*60,
               first_h=hits[0],second_h=hits[1],mode12_nm=stage_mode(spec),dark_retained=curve[dark][-1]/curve[dark][0],
               max_tail_cm3=curve.max(),initial_cm3=initial.sum(),final_cm3=j[-1].sum(),upper_out_cm3=b[1],clearing_cm3=b[5],entered_cm3=b[0],wall_cm3=b[2],diluted_cm3=b[3],coag_loss_cm3=b[4],
               number_residual=residual,cut_mapping_max_shift_min=np.nanmax(abs((crosses(phys_curve)-hits)*60)) if np.isfinite(hits).any() and np.isfinite(crosses(phys_curve)).any() else np.nan))
            for sid in range(9,19):
                mask=st==sid;valid=mask&~gap
                # Use fully observed channels downstream of the upper inlet anchor.
                err=np.abs(np.log10(1+spec[valid,6:])-np.log10(1+obs[valid,6:]))@np.diff(e)[6:]/np.diff(e)[6:].sum()
                stages.append(dict(**identifier,stage=sid,spectral_factor=10**err.mean(),tail_mae=np.mean(abs(curve[mask]-target[mask])),
                                last_cm3=curve[mask][-1],measured_last_cm3=target[mask][-1]))
            if not args.pilot and cells<288:continue
            trajectories.append(pd.DataFrame(dict(**identifier,archive_h=hours,stage=st,observed_tail=target,predicted_tail=curve,
                observed_7p7=obs[:,6]*np.diff(e)[6],predicted_7p7=spec[:,6]*np.diff(e)[6],predicted_exact_cut=phys_curve)))
    print(cells,len(rows),flush=True)
pd.DataFrame(rows).to_csv(args.out/'cases.csv',index=False)
pd.DataFrame(stages).to_csv(args.out/'stages.csv',index=False)
pd.concat(trajectories).to_csv(args.out/'trajectories.csv.gz',index=False,compression={'method':'gzip','mtime':0})
(args.out/'run.json').write_text(json.dumps(dict(cases=len(rows),seconds=time.monotonic()-start,device=wp.get_device('cuda:0').name,
    revision='af0dc67bf7',observed_first_second_h=observed_times.tolist(),observed_mode12_nm=stage_mode(obs),
    dark_observed=target[-1]/target[st==18][0],range_nm=[5.,96.],threshold_nm=9.7,wall_scale=wall_scale,blend=blend),indent=2))
