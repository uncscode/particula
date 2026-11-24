"""Replay one hypothesis with explicit acid/base molecular exchange and signed transport."""
import argparse,ctypes as ct,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import warp as wp
import chamber as ch
from molecular import simulate
from physics import PACKET,salt,amu,e,d,stage
cli=argparse.ArgumentParser()
cli.add_argument('config',type=Path);cli.add_argument('output',type=Path);cli.add_argument('lib',type=Path)
args=cli.parse_args();cfg=json.loads(args.config.read_text());out=args.output
out.mkdir(parents=True,exist_ok=True);tic=time.monotonic()
boxes=ch.boxes=cfg.get('boxes',64);ch.LIB=args.lib;vol=cfg.get('volume',.04)
seed=cfg.get('seed',418330);rng=np.random.default_rng(seed)
masscoef=1350*np.pi/6*1e-27/((salt+18.01528)*amu)
offset=masscoef*.8**3-1
def diameter(a):return ((offset+a)/masscoef)**(1/3)
states=np.arange(1,6);diam=diameter(states)
rates=np.vstack([np.zeros(5),np.column_stack([ch.ws*ch.wall(diam)+.0001754037,ch.charge_rates(diam)])])
def births(ac,s,air,config):
    neg,pos=air[:2]
    frac=.1*pos/(neg+.1*pos) if neg+.1*pos else 1/11
    neutral=ch.reaction(ac,False)*config.get('neutral_source',1.)
    ion=ch.reaction(ac,True) if s in (10,12,15,17) else 0.
    if config.get('split_power'):
        exp=config['split_power']
        neutral*=max(ac/6e7,1e-9)**(exp-3)
    if s in (10,12,15,17) and 'ion_share' in config:frac=config['ion_share']
    return np.array([ion*(1-frac)*config.get('negative_source',1.),neutral,ion*frac*config.get('positive_source',1.)])
if cfg.get('measured_acid'):
    measured=pd.read_csv(PACKET/'forcing.csv')
    good=np.isfinite(measured.measured_cal)
    replacement=np.interp(ch.hours,measured.archive_h[good],measured.measured_cal[good])
    if cfg['measured_acid']=='all':ch.gas=replacement
    molecular_acid=replacement
else:molecular_acid=ch.gas
stream,log=simulate(cfg,ch.seconds,ch.st,molecular_acid,ch.air,rates,births)
log.to_csv(out/'molecular.csv',index=False)
def transport(stream,start):
    counts=rng.poisson(stream[:,1,None]*stream[:,2:]*boxes*vol)
    ii,sign=np.nonzero(counts);ids=np.repeat(ii,counts[ii,sign]);signs=np.repeat(sign-1,counts[ii,sign]).astype('i4')
    born=stream[ids,0]+rng.random(len(ids))*stream[ids,1]
    period=(np.searchsorted(ch.seconds,born,side='right')-1).astype('i4')
    group=rng.integers(boxes,size=len(born),dtype='i4')
    seeds=rng.integers(1,2**63-1,len(born),dtype='uint64')
    end=masscoef*2.9**3
    grid=((start+np.arange(int(np.ceil(end-start))+1))/masscoef)**(1/3)
    c=ch.charge_rates(grid);c[:,0]*=cfg.get('attach',1);c[:,1:3]*=cfg.get('neutralize',1);c[:,3]*=cfg.get('field_scale',1)
    coef=np.column_stack([ch.rate(grid),ch.ws*ch.wall(grid)+.0001754037,c,ch.rate(grid),ch.rate(grid)])
    host=[ch.seconds,ch.gas,coef,ch.air,np.zeros((len(grid),3)),born,period,signs,seeds]
    if len(born):
        gpu=[wp.array(np.ascontiguousarray(v),device='cuda:0') for v in host]
        result=wp.empty((len(born),8),dtype=wp.float64,device='cuda:0');hist=wp.zeros((len(ch.seconds),2),dtype=wp.int32,device='cuda:0')
        lib=ct.CDLL(str(args.lib/'growth.so'))
        lib.clusters.argtypes=[ct.c_int]*4+[ct.c_double]*2+[ct.POINTER(ct.c_void_p)]
        wp.synchronize()
        assert lib.clusters(len(born),len(ch.seconds),1,0,start,end,(ct.c_void_p*11)(*[v.ptr for v in gpu+[result,hist]]))==0
        z=result.numpy();small=hist.numpy()
        np.testing.assert_allclose(z[:,2],start+z[:,4]-z[:,5])
    else:z=np.zeros((0,8));small=np.zeros((len(ch.seconds),2))
    return (born,signs,period,group,seeds),z,small
events,z,small=transport(stream,cfg.get('boundary',4)+1+offset)
n_early=len(z)
mature_input=np.zeros(3)
if np.any(cfg.get('mature_share',0.)):
    direct=stream.copy()
    for i,row in enumerate(direct):
        ti=row[0]+row[1]*.5
        j=np.searchsorted(ch.seconds,ti)-1
        k=j+int(ti>=(ch.seconds[j]+ch.seconds[j+1])/2)
        ac=np.interp(ti,ch.seconds,molecular_acid)
        intensity=cfg.get('contrast',0.) if ch.st[k] in (9,10,11,12,14,15,17) and not cfg.get('mean_births') else 0.
        rate=(births(ac*(1-intensity),ch.st[k],ch.air[k],cfg)+births(ac*(1+intensity),ch.st[k],ch.air[k],cfg))*.5
        row[2:]=rate*np.asarray(cfg['mature_share'])
    mature_input=(direct[:,1,None]*direct[:,2:]).sum(0)
    more,zz,ss=transport(direct,masscoef*1.7**3)
    zz[:,7]=more[0]
    events=tuple(np.concatenate(t) for t in zip(events,more))
    z=np.concatenate([z,zz]);small+=ss
born,signs,period,group,seeds=events
# Scale primary-ion rates consistently in the aerosol phase as well.
orig_rates=ch.charge_rates
def downstream_rates(diam):
    c=orig_rates(diam);c[:,0]*=cfg.get('attach',1);c[:,1:3]*=cfg.get('neutralize',1);c[:,3]*=cfg.get('field_scale',1)
    return c
ch.charge_rates=downstream_rates
res,audit,charged,qa=ch.population((born,signs!=0,period,group,seeds),z,vol,cfg.get('particle_dt',8),cfg.get('grid',384),1,183304,True)
scale=vol*boxes;bands=charged.sum(0);bands[:,0,:]+=small;bands/=scale
tails=res[:,:,-1];avg=tails.mean(0)
s17=z[z[:,7]>=0,7];s29=z[z[:,1]==1,0]
def flux(arr):return np.r_[np.nan,np.histogram(arr,ch.seconds)[0]/(np.diff(ch.seconds)*scale)]
df=pd.DataFrame(dict(archive_h=ch.hours,stage=ch.st,obs=ch.obs,tail=avg,flux17=flux(s17),flux29=flux(s29)))
signed=[]
for k,pol in enumerate(['negative','positive']):
    for b,span in enumerate(['2p2_to_2p9','2p9_to_5','5_to_9p7','9p7_to_40']):
        col=f'{pol}_{span}';df[col]=bands[:,b,k]
        raw=ch.ions_frame['ais_'+col+'_cm3']
        for sid in (10,11,12):
            v=pd.DataFrame(dict(i=ch.ions_frame.ais_native_row,y=raw,p=df[col]))[ch.st==sid].groupby('i').mean()
            signed.append(dict(stage=sid,polarity=pol,band=span,obs=v.y.median(),pred=v.p.median(),mae=(v.y-v.p).abs().mean()))
ion=pd.DataFrame(signed);ion.to_csv(out/'signed.csv',index=False);df.to_csv(out/'series.csv',index=False)
def crosses(a):
    return np.array([ch.hours[np.argmax(a>=v)] if (a>=v).any() else np.nan for v in (1000,5000)])
def maybe(x):return None if not np.isfinite(x) else float(x)
hits=crosses(avg);observed=crosses(ch.obs)
boot=np.array([crosses(tails[rng.integers(0,boxes,boxes)].mean(0)) for _ in range(500)])
rises=np.diff(boot,axis=1).ravel()*60;rises=rises[np.isfinite(rises)]
spec=res[:,:,:-1].mean(0)/np.diff(e);select=ch.st==12
stage_errors=[]
for sid in range(9,19):
    ok=(ch.st==sid)&~ch.missing[ch.ix]
    err=np.abs(np.log10(spec[ok,4:]+1)-np.log10(ch.y[ch.ix[ok],4:]+1))
    stage_errors.append(float(10**(err@np.diff(e)[4:]/np.diff(e)[4:].sum()).mean()))
stats=dict(config=cfg,boxes=boxes,run_seconds=time.monotonic()-tic,device=wp.get_device('cuda:0').name,
    source_cm3=log.source.sum()+mature_input.sum(),direct_source_cm3=mature_input.sum(),model_exit_cm3=log.exit.sum(),survival_molecular=float(log.exit.sum()/log.source.sum()) if log.source.sum() else None,
    embryos_sampled=len(born),exited17=len(s17),exited29=len(s29),peak=float(avg.max()),rise=maybe(np.diff(hits)[0]*60),
    errors_min=[maybe(x) for x in 60*(hits-observed)],spacing_95=np.quantile(rises,[.025,.975]).tolist() if len(rises) else [],
    fraction_boot_crossed=len(rises)/500,stage11_flux17=df.flux17[ch.st==11].mean(),stage11_flux29=df.flux29[ch.st==11].mean(),
    stage12_mode_nm=np.median(d[4:][np.argmax(spec[select,4:],axis=1)]),dark_retention=avg[-1]/max(1.,avg[ch.st==18][0]),
    ion_stage10_12_small_mae=ion.query("stage!=11 and band in ['2p2_to_2p9','2p9_to_5']").mae.mean(),
    error_factor_by_stage=stage_errors,mean_ammonia_per_cluster=float(log.ammonia_exposure.sum()/log.exposure.sum()) if log.exposure.sum() else None,
    max_sulfate_residual=audit[:,0].max(),max_signed_residual=float(abs(qa[:,0]).max()),relative_number_error=abs(log.number_error.sum())/max(1,log.source.sum()))
(out/'result.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
print(cfg['name'],stats['rise'],stats['errors_min'],stats['peak'],stats['ion_stage10_12_small_mae'],flush=True)
