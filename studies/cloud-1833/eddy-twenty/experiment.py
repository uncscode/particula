"""Sequential chamber experiments. Each specification is frozen before running it."""
import ctypes as ct
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import warp as wp
from treatments import coefficient, tag_inlet, REFERENCE_HOUR
from domain import geometry
from boundary import blend,growth
from physics import PACKET,d,e,t,y,stage,missing
HERE=Path(__file__).resolve().parent
obs=pd.read_csv(PACKET/'forcing.csv')
pre=pd.read_csv(PACKET/'five-nm-flux/boundary.csv')
beam=np.isin(stage,[1,2,3,4,5,6,9,10,11,12,14,15,17])
windows={'X':(9,18),'H':(1,7),**{str(i):(i,i) for i in [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]}}
def parts(k):
    return np.flatnonzero((stage>=windows[k][0])&(stage<=windows[k][1]))
bands=np.array([7.7,9.7,15.,25.,40.,10**e[-1]])
def overlaps(L,U):
    return np.maximum(0,np.minimum(e[1:],np.log10(U))-np.maximum(e[:-1],np.log10(L)))
bm=np.array([overlaps(l,u) for l,u in zip(bands,bands[1:])]).T
tail=bm[:,1:].sum(1)
lib=ct.CDLL(sys.argv[1]);lib.replay.argtypes=[ct.c_int]*3+[ct.c_double]*2+[ct.POINTER(ct.c_void_p)]
out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
cache={}
def boundary(cfg,ids):
    key=cfg.get('inlet','direct')
    if np.all(np.isin(stage[ids],range(9,19))):
        pos=np.searchsorted(pre.archive_h,t[ids]);f=pre[key].to_numpy()[pos].copy()
    else:
        if key!='direct':raise ValueError('Alternative inlet unavailable for this segment')
        if 'vel5' not in cache:
            a,b=growth(np.array([5.]));cache['vel5']=(a+blend*(b-a))[0]
        mix=np.log(5/d[4])/np.log(d[5]/d[4])
        f=obs.measured_cal.to_numpy()[ids]*cache['vel5']*((1-mix)*y[ids,4]+mix*y[ids,5])
    return f*cfg.get('inlet_scale',1.)
def calc(configs,window,n=192):
    ids=parts(window);h=t[ids];nt=len(h);st=stage[ids];count=len(configs)
    g=geometry(n,'aiomfac',1.);D=g['D'];edge=10**g['edges'];hard,attr=g['vel']
    initial=g['mi']@y[ids[0]];gas=[];wall=[];speed=[];inputs=[];settings=[];tags=[]
    if any(c.get('tag_all') for c in configs):
        assert all(c.get('tag_all') and c.get('coag')==0 for c in configs)
        initial=initial*0
    for c in configs:
        acid=obs[c.get('acid','measured_cal')].to_numpy()[ids].copy()
        acid*=np.where(np.isin(st,[7,13,16,18]),c.get('dark_acid',1.),1.)
        if 'constant_acid' in c:acid.fill(c['constant_acid'])
        rate=coefficient(D,c,g['wall'])*(D/15.)**c.get('tilt',0.)
        rate*=np.where(D<9.7,c.get('low_wall',1.),1.)
        rate+=c.get('flat_loss',0.)
        cap=(hard+blend*(attr-hard)).copy()
        cutoff=c.get('cut',9.7)
        mask=edge<cutoff
        cap[mask]=hard[mask]+c.get('attraction',blend)*(attr-hard)[mask]
        cap[mask]*=c.get('low_capture',1.)
        cap[~mask]*=c.get('high_capture',1.)
        f=boundary(c,ids)
        if 'shift_min' in c:f=np.interp(h-c['shift_min']/60,h,f)
        if 'early_factor' in c:f*=np.where(st<=10,c['early_factor'],1.)
        if 'mid_factor' in c:f*=np.where(st==11,c['mid_factor'],1.)
        if 'late_factor' in c:f*=np.where(st==12,c['late_factor'],1.)
        tag=tag_inlet(h,c)
        if 'remove_offset' in c:
            subtract=tag_inlet(h,c|{'offset':c['remove_offset']})
            np.testing.assert_allclose(np.trapezoid(subtract,h*3600),np.trapezoid(tag,h*3600),atol=1e-9)
            f-=subtract
        f=f+tag
        assert f.min()>=0
        if c.get('tag_all'):tag=f.copy()
        tags.append(tag)
        gas.append(acid);wall.append(rate);speed.append(np.r_[cap,cap.max()]);inputs.append(f)
        settings.append([1.1,1.,float(c.get('inlet','').startswith('prior_')),1.,.0001754037*c.get('dilution',1.),c.get('coag',1.)])
    host=[h,gas,np.ones(nt),initial,speed,wall,settings,g['K'],*g['pairs'],g['over'],np.zeros(nt),inputs]
    gpu=[wp.array(np.ascontiguousarray(v),device='cuda:0') for v in host]
    x=wp.empty((count,nt,n),dtype=wp.float64,device='cuda:0');bb=wp.empty((count,n,6),dtype=wp.float64,device='cuda:0')
    tin=wp.array(np.ascontiguousarray(tags),device='cuda:0');tout=wp.empty_like(x);tb=wp.empty((count,n,2),dtype=wp.float64,device='cuda:0')
    wp.synchronize();ptr=(ct.c_void_p*20)(*[u.ptr for u in gpu+[x,bb,tin,tout,tb]])
    assert lib.replay(count,n,nt,g['dx'],12 if n<=192 else 6,ptr)==0
    state=x.numpy();book=bb.numpy().sum(1);tr=tout.numpy();tbook=tb.numpy().sum(1)
    assert np.isfinite(tr).all() and tr.min()>=-1e-9
    assert np.isfinite(state).all() and state.min()>-1e-9
    frames=[];metrics=[]
    measured=y[ids]@bm;mt=y[ids]@tail;spec=state@(g['mo']/np.diff(e)[:,None]).T
    for z,(c,b) in enumerate(zip(configs,book)):
        if c.get('tag_all'):
            np.testing.assert_allclose(tr[z],state[z],atol=1e-9,rtol=1e-9)
        expected=np.trapezoid(tags[z],h*3600)
        np.testing.assert_allclose(tbook[z,0],expected,atol=1e-8)
        tag_err=abs(tr[z,-1].sum()+tbook[z,1]-expected)/max(1,expected)
        assert tag_err<1e-9
        projected=tr[z]@(g['mo']/np.diff(e)[:,None]).T
        f=np.array(inputs[z]);dt=np.diff(h)*3600
        supplied=f[:-1]@dt if settings[z][2] else np.trapezoid(f,h*3600)
        np.testing.assert_allclose(supplied,b[0],rtol=1e-9,atol=1e-7)
        err=abs(state[z,-1].sum()-initial.sum()-b[0]+b[1:].sum())/max(1,initial.sum()+supplied)
        assert err<1e-9
        v=spec[z]@bm;curve=spec[z]@tail
        frame=pd.DataFrame({'archive_h':h,'stage':st,'predicted':curve,'observed':mt,'flux':f})
        for label,(lo,hi) in {'small':(7.7,9.7),'middle':(9.7,12.),'tail':(9.7,10**e[-1])}.items():
            weights=overlaps(lo,hi)
            frame['tag_'+label]=projected@weights
            frame['model_'+label]=spec[z]@weights
            frame['obs_'+label]=y[ids]@weights
        for k in range(5):
            frame[f'p{k}']=v[:,k];frame[f'o{k}']=measured[:,k]
        frame.attrs["channel_counts"]=(spec[z]*np.diff(e),y[ids]*np.diff(e))
        frames.append(frame)
        def crossing(s,val):
            hits=np.flatnonzero(s>=val)
            return float(h[hits[0]]) if len(hits) else np.nan
        a1,a5=[crossing(curve,i) for i in (1000,5000)];b1,b5=[crossing(mt,i) for i in (1000,5000)]
        mode=np.median(d[6:][np.argmax(spec[z,st==12,6:],axis=1)]) if np.any(st==12) else np.nan
        dark=st==18
        metrics.append(dict(tag_dose=expected,tag_error=tag_err,tag_at_crossing=frame.tag_tail.iloc[np.argmin(abs(h-REFERENCE_HOUR))],budget_residual=err,source=supplied,initial=initial.sum(),final=state[z,-1].sum(),wall=b[2],dilution_loss=b[3],merger_loss=b[4],upper=b[1],
            first_min=(a1-b1)*60,second_min=(a5-b5)*60,rise_min=(a5-a1)*60,mode12=mode,
            retention=curve[dark][-1]/max(1,curve[dark][0]) if dark.any() else np.nan))
    return frames,metrics
def loss(df,rule):
    a=df.iloc[1:].copy()
    if 'stages' in rule:a=a[a.stage.isin(rule['stages'])]
    if 'minutes' in rule:
        minutes=(a.archive_h-df.archive_h.iloc[0])*60
        a=a[(minutes>=rule['minutes'][0])&(minutes<=rule['minutes'][1])]
    ks=rule.get('bands',[1,2,3,4]);p=a[[f'p{k}' for k in ks]].to_numpy();o=a[[f'o{k}' for k in ks]].to_numpy()
    floor=rule.get('floor',25)
    if 'channels_nm' in rule:
        lo,hi=rule['channels_nm']; sel=(d>=lo)&(d<=hi); pred,obs=df.attrs['channel_counts']
        p=pred[a.index][:,sel];o=obs[a.index][:,sel]
        outage=missing[np.searchsorted(t,a.archive_h.to_numpy())]
        invalid=outage[:,None] & (np.flatnonzero(sel)<7)[None,:]
        if invalid.any():p,o=p[~invalid],o[~invalid]
    assert p.size and np.isfinite(p).all() and np.isfinite(o).all()
    return float(np.abs(np.log((p+floor)/(o+floor))).mean())
def experiment(spec):
    tic=time.monotonic();folder=out/spec['id'];folder.mkdir(exist_ok=True)
    (folder/'proposal.json').write_text(json.dumps(spec,indent=2)+'\n')
    base=spec.get('base',{}).copy()
    if 'parent' in spec:base=json.loads((out/spec['parent']/'winner.json').read_text())|base
    axes=spec['grid'];cs=[base|dict(zip(axes,vals)) for vals in itertools.product(*axes.values())]
    if 'variants' in spec:cs=[base|v for v in spec['variants']]
    n=spec.get('cells',192);profiles=[];cost=np.zeros(len(cs));rules=spec.get('fit',[{'window':'18'}]);solves=0
    for r in rules:
        fs,met=calc(cs,r['window'],n);solves+=len(cs)
        scores=np.array([loss(f,r) for f in fs]);cost+=scores/len(rules)
        for k,m in enumerate(met):profiles.append(dict(candidate=k,role='fit',window=r['window'],loss=scores[k],**m))
    winner=int(np.argmin(cost))
    if spec.get('choice') is not None:winner=spec['choice']
    (folder/'winner.json').write_text(json.dumps(cs[winner],indent=2)+'\n')
    windows=spec.get('test',['X']);curves=[]
    for w in windows:
        fs,met=calc(cs,w,n);solves+=len(cs)
        for k,m in enumerate(met):
            profiles.append(dict(candidate=k,role='test',window=w,loss=loss(fs[k],{}),**m))
            if k==winner or len(cs)<=6 or spec.get('keep_all',False):
                curve=fs[k].assign(candidate=k,window=w);curve.attrs={}
                curves.append(curve)
    profile=pd.DataFrame(profiles)
    profile.to_csv(folder/'runs.csv',index=False)
    pd.DataFrame([dict(candidate=i,cost=cost[i],config=json.dumps(c,sort_keys=True)) for i,c in enumerate(cs)]).to_csv(folder/'candidates.csv',index=False)
    pd.concat(curves).to_csv(folder/'series.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    info=dict(cases=solves,device=wp.get_device('cuda:0').name,seconds=time.monotonic()-tic,winner=winner,cells=n)
    (folder/'execution.json').write_text(json.dumps(info,indent=2))
    table=profile.query("role=='test' and window=='X'").copy()
    print(spec['id'],len(cs),'best',winner,cs[winner],flush=True)
    print(table.loc[table.candidate==winner,['first_min','second_min','retention','tag_at_crossing']].round(3).to_string(index=False),flush=True)
    print('READY',flush=True)
if __name__=='__main__':
    if len(sys.argv)>3:
        for design in json.loads(Path(sys.argv[3]).read_text()):
            experiment(design)
    else:
        print('READY',flush=True)
        for line in sys.stdin:
            if line.strip()=='STOP':break
            if line.strip():experiment(json.loads(line))
