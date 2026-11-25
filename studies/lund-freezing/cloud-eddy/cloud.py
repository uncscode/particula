# %%
"""Run a particle-resolved, prescribed eddy in a Particula finite-volume grid on CUDA."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
import particula
from particula.coupling import FlowGrid
from particula.coupling.flows import GridInterval
from particula.coupling.flow_replay import GridReplay
from particula.coupling.grid_run import GridRun
from particula.coupling.grid_inventory import InventoryLayout
from particula.coupling.grid_warp import transport

import physics as gpu

L,H=1600.,600.
ORIGINS=('resident','fresh')
BASE=Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class Case:
    name:str='fresh'
    n:int=10
    per:int=48   # each of two represented populations per 100 reference cells
    dt:float=10.
    hours:float=6.
    entrain:float=.09
    eddy:float=.55
    rh:float=.85
    source:str='fresh'
    law:str='fixed'
    reset:int=0
    beta:float=.5
    seed:int=20261125
    fall:float=1.
    active_share:float=1.
    plate:int=0
    mixing:float=0.
    loading:float=1.
    matched_closed:bool=False


def rate_input(packet):
    sys.path.insert(0,str(packet))
    from study import read_inputs,geometry,train
    g=geometry(read_inputs(packet,False)[0]);g=g[g['sample']=='MineralDust']
    rows=[]
    for law,b in [('fixed',1.),('aging',.6)]:
        _,rates,prior,_,post=train(g,power=b)
        for plate,night in g[(g.phase=='night')&(g.h>0)].groupby('plate'):
            ids=(plate-1)*100+night.id-1
            weights=post[ids].copy();weights[night['shift']>80]=prior
            fresh=weights.mean(axis=0)
            old=fresh*np.exp(-rates*2**b);old/=old.sum()
            rows.extend(dict(law=law,plate=plate,rate=r,fresh=f,aged=o) for r,f,o in zip(rates,fresh,old))
    return pd.DataFrame(rows)


def geometry(c, base_temperature=-11.6, cloud_base=240., profile=None):
    n=c.n
    grid=FlowGrid((n,1,n),((0,L),(0,1.),(0,H)),('periodic','periodic','closed','closed','closed','open'))
    dx,dz=L/n,H/n;vol=dx*dz
    x,z=np.meshgrid(np.linspace(0,L,n+1),np.linspace(0,H,n+1),indexing='ij')
    k=2*np.pi/L;eta=z/H
    psi=-np.sin(k*x)/k*(c.eddy*np.sin(np.pi*eta)+c.entrain*eta**3)
    psi[-1]=psi[0];psi[:,0]=0.
    if c.entrain==0:psi[:,-1]=0.
    fx=np.diff(psi,axis=1);fz=-np.diff(psi,axis=0)
    flux=np.zeros((n,n,6))
    flux[:,:,0]=-fx[:-1];flux[:,:,1]=fx[1:]
    flux[:,:,4]=-fz[:,:-1];flux[:,:,5]=fz[:,1:]
    assert np.max(abs(flux.sum(2)))<1e-10
    xyz=grid.coordinates;zc=xyz[:,2]
    pressure=np.full(len(zc),88000.)
    tc=base_temperature-.006*zc
    if profile is not None:
        tc,pressure,vapor,liquid=profile(zc)
    t=tc+273.15
    # Murphy & Koop (2005), saturation over supercooled water and planar ice.
    lew=(54.842763-6763.22/t-4.210*np.log(t)+.000367*t+
         np.tanh(.0415*(t-218.8))*(53.878-1331.22/t-9.44523*np.log(t)+.014025*t))
    ei=np.exp(9.550426-5723.265/t+3.53068*np.log(t)-.00728332*t)
    es=np.column_stack([np.exp(lew),ei]);sat=es/(461.5*t[:,None])
    dv=2.12e-5*(t/273.15)**1.94*101325/pressure
    G=[]
    for j,latent,rho in [(0,2.5e6,1000),(1,2.834e6,917)]:
        G.append(1/(rho*(461.5*t/(dv*es[:,j])+latent/(.023*t)*(latent/(461.5*t)-1))))
    sat=np.column_stack([sat,*G,tc])
    if profile is None:
        qtotal=np.interp(cloud_base,zc[:n],sat[:n,0])
        vapor=np.minimum(qtotal,sat[:,0])
        liquid=np.maximum(0,qtotal-vapor)
    # Prescribed-flow metadata carry vapor per unit dry-air mass.
    rho_dry=1.1 if profile is None else (pressure-vapor*461.5*t)/(287.05*t)
    air=np.column_stack([np.zeros((len(zc),3)),t,pressure,vapor/rho_dry])
    speed=np.zeros((len(zc),3)); xc=xyz[:,0]; eta=zc/H
    speed[:,0]=-np.sin(k*xc)*(c.eddy*np.pi*np.cos(np.pi*eta)+3*c.entrain*eta**2)/(k*H)
    speed[:,2]=np.cos(k*xc)*(c.eddy*np.sin(np.pi*eta)+c.entrain*eta**3)
    air[:,:3]=speed
    swept=c.dt*flux.reshape(-1,6)
    assert np.max(np.maximum(swept,0).sum(1))/vol<.8
    mix=np.zeros_like(swept)
    for side in (0,1,4,5):
        spacing=dx if side<2 else dz
        mix[:,side]=np.where(grid.neighbors[:,side]>=0,c.mixing*c.dt*vol/spacing**2,0.)
    assert np.max((np.maximum(swept,0)+mix).sum(1))/vol<.8
    frame=GridInterval(0,c.dt,air,swept,mix,speed,np.tile([c.mixing,0.,c.mixing],(n*n,1)))
    return grid,GridReplay(grid,[frame],'analytic streamfunction and dry cloud-top exchange'),sat,vapor,liquid


def populations(c,rates,liquid,background_ccn=50.,inlet_loading=None):
    # Stratified initial sampling and a flux-weighted continuous stream of arrivals.
    rng=np.random.default_rng(c.seed); count=c.per*100; seconds=int(c.hours*3600)
    added=int(round(count*seconds*c.entrain/(np.pi*H)))
    cells=np.repeat(np.arange(100),c.per)
    x0=(cells//10+rng.random(count))*L/10
    z0=(cells%10+rng.random(count))*H/10
    xf=(np.pi-np.arcsin(1-2*rng.random(added)))*L/(2*np.pi)
    births=(np.arange(added)+.5)*seconds/max(added,1)
    x=np.tile(np.r_[x0,xf],2);z=np.tile(np.r_[z0,np.full(added,H-1e-7)],2)
    bt=np.tile(np.r_[np.full(count,-1.),births],2)
    car=np.repeat([0,1],count+added)
    a=np.zeros(len(x),dtype=gpu.State.numpy_dtype())
    a['pos'][:,0]=x;a['pos'][:,1]=z;a['birth']=bt;a['fate']=(bt<0)
    a['cell']=((x*c.n/L).astype(int)*c.n+(z*c.n/H).astype(int))
    a['carrier']=car
    a['weight']=np.where(car==1,1000.*c.loading,background_ccn*1e6)*L*H/count
    if inlet_loading is not None:
        a['weight'][(bt>=0)&(car==1)]*=inlet_loading/c.loading
    if added:
        a['weight'][bt>=0]*=(count*seconds*c.entrain/(np.pi*H))/added
    dry=np.clip(rng.lognormal(np.log(65e-9),.32,len(x)),25e-9,200e-9)
    a['dry_radius']=dry
    # Initialize exactly the prescribed bulk mass by cell, even at finer resolution.
    resident=bt<0; volume=L*H/c.n**2
    sums=np.bincount(a['cell'][resident],weights=a['weight'][resident],minlength=c.n**2)
    assert (sums>0).all()
    a['mass'][resident]=liquid[a['cell'][resident]]*volume/sums[a['cell'][resident]]
    a['phase']=(a['mass']>=4*np.pi/3*1000e-18);a['wets']=a['phase']
    group=rates[rates.law==c.law]
    if c.plate:group=group[group.plate==c.plate]
    table=group.groupby('rate')[['fresh','aged']].mean()
    u=rng.random(len(x))
    for tag,which in [(False,'aged'),(True,'fresh' if c.source=='fresh' else 'aged')]:
        mask=(car==1)&((bt>=0)==tag)
        a['rate'][mask]=table.index.to_numpy()[np.searchsorted(table[which].cumsum().to_numpy()/table[which].sum(),u[mask])]
    a['rate'][(bt>=0)&(rng.random(len(x))>=c.active_share)]=0.
    if c.source=='inert':a['rate'][bt>=0]=0
    if c.source=='off':a['rate']=0
    if c.law=='aging':
        a['lastwet'][(bt<0)|((bt>=0)&(c.source=='aged'))]=7200.
    a['threshold']=rng.exponential(size=len(x));a['first']=-1
    return a


def stats(c,a,v,imp,exp,restore,t,initial_mass):
    vdom=L*H; grams_to_path=1000/L; mult=a['weight']
    active=a['fate']==1; born=a['fate']!=0; fresh=a['birth']>=0; carriers=a['carrier']==1
    mass=a['mass']*mult
    def rate(values):
        return values.sum()/vdom/1000
    ice=active&(a['phase']==2);liq=active&~ice
    accounted=v.sum()+mass[born].sum()+exp.sum()-imp.sum()-restore.sum()
    assert abs(accounted-initial_mass)<2e-8*initial_mass, (accounted,initial_mass)
    assert np.all(np.isfinite(a['mass'])) and min(a['mass'].min(),v.min())>=-1e-16
    assert (a['freezes']>=a['sublimations']).all()
    # Every activated record is airborne, exported or deposited, with no lost IDs.
    assert np.array_equal(born,a['birth']<t)
    for group in [carriers,~carriers]:
        expected_count=mult[group&(a['birth']<t)].sum()
        accounted_count=sum(mult[group&(a['fate']==fate)].sum() for fate in (1,2,3))
        assert np.isclose(accounted_count,expected_count,rtol=1e-12,atol=0)
    assert np.array_equal(a['freezes']>0,a['first']>=0)
    record=dict(t_h=t/3600,primary=rate(mult*a['freezes']),unique=rate(mult*(a['first']>=0)),
                suspended=rate(mult[ice]),LWP=mass[liq].sum()*grams_to_path,IWP=mass[ice].sum()*grams_to_path,
                evaporated=(mult*a['evap']).sum()*grams_to_path,condensed=(mult*a['cond']).sum()*grams_to_path,
                deposited_vapor=(mult*a['dep']).sum()*grams_to_path,sublimed=(mult*a['sub']).sum()*grams_to_path,
                fallout_mass=mass[a['fate']==3].sum()*grams_to_path,
                relative_water_error=(accounted-initial_mass)/initial_mass,
                birth_count=int(born.sum()),water_import_kg=imp.sum(),water_export_kg=exp.sum(),
                surface_water_kg=restore.sum())
    for label,mask in [('resident',~fresh),('fresh',fresh)]:
        cc=mask&carriers
        record[label+'_primary']=rate((mult*a['freezes'])[cc])
        record[label+'_recycled_primary']=rate((mult*a['recycled_ice'])[cc])
        record[label+'_rewetted']=rate(mult[cc&(a['wets']>=2)])
        record[label+'_inflow_or_initial']=rate(mult[cc&born])
        record[label+'_live_ice']=rate(mult[cc&ice])
        record[label+'_sublimation_events']=rate((mult*a['sublimations'])[cc])
    return record


def run(c,rates,out,detail=False,trace_ids=None, *, base_temperature=-11.6,
        cloud_base=240., closed_entrain=.09, endpoint=None, treatment=None, profile=None,
        inlet_density=None, lower_restore=True, background_ccn=50., flow=None, lower_source=None, inlet_loading=None):
    wp.init();dev=wp.get_device('cuda:0');assert dev.is_cuda
    grid,forcing,sat,vapor,liquid=geometry(c,base_temperature,cloud_base,profile);n=c.n;vol=grid.volume
    inlet=np.zeros((n*n,6,1));inlet[:,5,0]=c.rh*sat[:,0] if inlet_density is None else inlet_density
    run=GridRun(forcing,InventoryLayout(('water',)),(vapor*vol)[:,None],inlet=inlet,device='cuda:0')
    inventory_case=dataclasses.replace(c,entrain=closed_entrain) if c.matched_closed else c
    a=populations(inventory_case,rates,liquid,background_ccn,inlet_loading)
    if c.matched_closed:
        a['birth'][a['birth']>=0]=1e12   # never release, resident IDs/draws are preserved
    if treatment is not None:
        a=treatment.initialize(a,sat,dev)
    s=wp.array(a,dtype=gpu.State,device=dev)
    initial=(vol*vapor).sum()+(a['mass']*a['weight']).sum()
    arr=lambda x:wp.array(np.ascontiguousarray(x),dtype=wp.float64,device=dev)
    zs=lambda:wp.zeros(n*n,dtype=wp.float64,device=dev)
    total,scale,restore=zs(),zs(),zs()
    clock=arr(np.array([0.]));satgpu=arr(sat)
    q=forcing.frames[0].transfer
    incoming=arr(np.maximum(-q,0)+forcing.frames[0].mixing);outgoing=arr(np.maximum(q,0)+forcing.frames[0].mixing)
    launch=lambda kernel,dim,args:wp.launch(kernel,dim,inputs=args,device=dev)
    if flow is not None:
        flow.bind(c,grid,forcing.frames[0],dev)
    if lower_source is not None:
        wp.load_module(module=lower_source[0].module,device=dev)
    def step(src,dst):
        if flow is not None:
            flow.faces(clock,c.dt,incoming,outgoing)
        launch(transport,(n*n,1),[run.neighbor,incoming,outgoing,run.volume,run.inlet,src,dst,run.imports,run.exports])
        if lower_restore:
            launch(gpu.local_air,n*n,[dst,total,satgpu,clock,c.dt,n,vol[0],restore])
        if lower_source is not None:
            launch(lower_source[0],n,[dst,restore,n,lower_source[1]*L/n*c.dt])
        if flow is None:
            launch(gpu.move,len(a),[s,clock,c.dt,n,c.eddy,c.entrain,c.fall])
        else:
            flow.move(s,clock,c)
        if c.mixing:
            launch(gpu.disperse,len(a),[s,clock,c.dt,n,c.mixing,c.seed])
        for ice in (1,0):
            launch(gpu.propose,len(a),[s,dst,satgpu,vol[0],c.dt,clock,ice,total])
            launch(gpu.limit,n*n,[dst,satgpu,total,scale,vol[0],ice])
            launch(gpu.phase_step,len(a),[s,scale,satgpu,clock,c.dt,ice,c.beta,
                                        1. if c.law=='fixed' else .6,c.reset,c.seed])
        if treatment is not None:
            treatment.freeze(s,clock,c.dt)
        launch(gpu.tick,1,[clock,c.dt])
    # Dry-run compilation by explicit module loading before CUDA graph capture.
    for module in [gpu.move.module,transport.module]:
        wp.load_module(module=module,device=dev)
    with wp.ScopedCapture(device=dev) as cap:
        step(run.state,run.scratch);step(run.scratch,run.state)
    if trace_ids is not None:
        select=wp.array(trace_ids,dtype=wp.int32,device=dev)
        traced=wp.empty(len(trace_ids),dtype=gpu.State,device=dev)
    def snapshot(t):
        a=s.numpy();rec=stats(c,a,run.state.numpy(),run.imports.numpy(),run.exports.numpy(),restore.numpy(),t,initial)
        return a,dict(case=c.name,seed=c.seed,**rec)
    a0,r0=snapshot(0);hist=[r0];samples=[];fields=[]
    count=int(round(c.hours*3600/c.dt));assert count%2==0
    tick_interval=round(600/(2*c.dt)); assert tick_interval>0
    begin=time.monotonic();paths=[]
    for j in range(1,count//2+1):
        wp.capture_launch(cap.graph)
        if trace_ids is not None:
            launch(gpu.sample_ids,len(trace_ids),[s,select,traced])
            selected=traced.numpy()
            for idx,record in zip(trace_ids,selected):
                paths.append((j*2*c.dt,idx,*record['pos'],record['birth'],record['first'],
                              record['fate'],record['phase'],record['wets'],record['dries'],
                              record['freezes'],record['mass']))
        if j%tick_interval==0 or j==count//2:
            t=j*2*c.dt;av,r=snapshot(t);hist.append(r)
            if detail:
                # Spatial fields from physical multiplicity and individual current masses.
                f=av['fate']==1;ids=av['cell'][f];m=av['weight'][f]*av['mass'][f]
                water=np.bincount(ids,weights=m*(av['phase'][f]!=2),minlength=n*n)/vol*1000
                ni=np.bincount(ids,weights=av['weight'][f]*(av['phase'][f]==2),minlength=n*n)/vol/1000
                for k in range(n*n):
                    fields.append((t,k,*grid.coordinates[k,[0,2]],water[k],ni[k]))
                samples.append(av[['pos','fate','phase','wets','dries','freezes','mass']].copy())
    wp.synchronize_device(dev)
    dur=time.monotonic()-begin
    if endpoint is not None:
        endpoint(av)
    if trace_ids is not None:
        pd.DataFrame(paths,columns=['time_s','id','x_m','z_m','birth_s','first_ice_s','fate',
                                    'phase','wet_entries','dryings','freezes','water_kg']).to_csv(
                                        out/'tagged_paths.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    if detail:
        out.mkdir(parents=True,exist_ok=True)
        # Save trajectories for actual IDs selected by outcome and source, plus all freezing IDs.
        ids=np.arange(len(av))
        frozen=av['first']>=0;src=np.where(av['birth']<0,'resident','incoming')
        ev=pd.DataFrame(dict(id=ids[frozen],source=src[frozen],birth_s=av['birth'][frozen],
                             first_freeze_s=av['first'][frozen],x_m=av['firstx'][frozen],z_m=av['firstz'][frozen],
                             multiplicity=av['weight'][frozen],freezes=av['freezes'][frozen],
                             recycled_freezes=av['recycled_ice'][frozen],final_fate=av['fate'][frozen]))
        ev.to_csv(out/'freezing_ids.csv',index=False)
        chosen=np.unique(np.concatenate([ids[mask][:6] for mask in
                           [frozen&(src==tag) for tag in ORIGINS[:1]]+
                           [frozen&(src=='incoming'),(av['freezes']>0)&(av['dries']>0),
                            (av['carrier']==1)&(av['wets']>=2)&(av['birth']>=0)]]))
        trajectories=[]
        for snapshot_id,ss in enumerate(samples):
            for i in chosen:
                aa=ss[i];trajectories.append((hist[snapshot_id+1]['t_h'],i,src[i],*aa['pos'],
                                      aa['fate'],aa['phase'],aa['wets'],aa['dries'],aa['freezes'],aa['mass']))
        pd.DataFrame(trajectories,columns=['t_h','id','source','x','z','fate','phase','wet_entries','dryings','freezes','water_kg']).to_csv(out/'tracks.csv',index=False)
        pd.DataFrame(fields,columns=['t_s','cell','x','z','liquid_g_m3','ice_L']).to_csv(out/'fields.csv',index=False)
    return hist,dict(**dataclasses.asdict(c),records=len(a),GPU=dev.name,seconds=dur,steps=count,
                     max_water_error=max(abs(r['relative_water_error']) for r in hist))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('packet',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--stage',choices=['pilot','main','checks','plates','source','refine','tracks','environment','ventilation','fine_controls'],required=True)
    args=ap.parse_args();p=args.out;p.mkdir(exist_ok=True,parents=True)
    rate=rate_input(args.packet);rate.to_csv(p/'rates.csv',index=False)
    default=Case()
    replace=dataclasses.replace
    pairs=[default,replace(default,name='aged',source='aged')]
    if args.stage=='pilot':
        cases=[replace(c,per=12,hours=.5) for c in pairs]
    elif args.stage=='main':
        cases=[replace(c,seed=default.seed+k) for k in range(8) for c in pairs]
        cases+= [replace(default,name=name,**opt) for name,opt in
                 [('inert',dict(source='inert')),('closed',dict(entrain=0.)),('off',dict(source='off'))]]
    elif args.stage=='checks':
        scenarios=[('timestep',dict(dt=5)),('grid',dict(n=20,dt=5,per=96)),('particles',dict(per=96)),
                   ('flat_rate',dict(beta=0.)),('dry',dict(rh=.70)),('weak_flow',dict(eddy=.3)),
                   ('aging_pause',dict(law='aging')),('aging_reset',dict(law='aging',reset=1))]
        cases=[replace(c,name=s+'_'+c.name,**opts) for s,opts in scenarios for c in pairs]
    elif args.stage=='fine_controls':
        cases=[replace(default,n=n,dt=dt,per=per,name=f'{label}_{n}',**kwargs)
               for n,per,dt in [(20,192,5.),(40,768,2.5)]
               for label,kwargs in [('closed',dict(entrain=0.,matched_closed=True)),('saturated',dict(rh=1.))]]
    elif args.stage=='ventilation':
        cases=[replace(default,name=label,seed=default.seed+k,**opts) for k in range(8)
               for label,opts in [('closed_matched',dict(entrain=0.,matched_closed=True)),
                                  ('saturated_inlet',dict(rh=1.)),('no_freezing',dict(source='off'))]]
    elif args.stage=='environment':
        choices=[('mixing',dict(mixing=2.)),('low_loading',dict(loading=.1)),('high_loading',dict(loading=10.))]
        cases=[replace(c,name=label+'_'+c.name,seed=c.seed+i,**opts)
               for i in range(4) for label,opts in choices for c in pairs]
    elif args.stage=='refine':
        variants=[('twenty',dict(n=20,per=192,dt=5)),('thirty',dict(n=30,per=432,dt=2.5)),
                  ('forty',dict(n=40,per=768,dt=2.5)),('moist',dict(rh=1.)),
                  ('slow_fall',dict(fall=.5)),('small_dt',dict(dt=2.5))]
        cases=[replace(c,name=s+'_'+c.name,**opts) for s,opts in variants for c in pairs]
    elif args.stage=='source':
        cases=[replace(default,name=label,active_share=f,seed=default.seed+k)
               for k in range(8) for label,f in [('inactive',0.),('quarter',.25),('third',1/3),('half',.5)]]
    elif args.stage=='tracks':
        cases=[default]
    else:
        cases=[replace(c,name=f'plate{plate}_'+c.name,plate=plate) for plate in range(1,6) for c in pairs]
    h=[];meta=[]
    for c in cases:
        ids=None
        if args.stage=='tracks':
            events=pd.read_csv(p/'freezing_ids.csv')
            selected=[]
            for tag in ['resident','incoming']:
                g=events[(events.source==tag)&(events.recycled_freezes>0)].sort_values('first_freeze_s')
                selected.extend(g.iloc[[len(g)//4,len(g)//2,3*len(g)//4]].id)
            ids=np.asarray(selected,dtype=np.int32)
        series,info=run(c,rate,p,detail=args.stage=='main' and c==default,trace_ids=ids)
        h.extend(series);meta.append(info)
        print(c.name,c.seed,round(info['seconds'],2),[(x,round(series[-1][x],5)) for x in
              ['primary','resident_primary','fresh_primary','LWP','IWP']],flush=True)
        pd.DataFrame(h).to_csv(p/(args.stage+'_history.csv'),index=False)
        (p/(args.stage+'_runs.json')).write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':
    main()
