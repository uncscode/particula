# %%
"""Paired CUDA cloud experiments using measured hold-entry rate distributions."""
import argparse
import gzip
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
import math
import os
import signal
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent
sys.path[:0] = [str(PACKET / 'cloud-eddy'), str(PACKET)]
import cloud
from study import SAMPLES, geometry, read_inputs, train

import isdac
import mpace
from loading import loadings
from minerals import Mineral
from unsteady import Roll, supply

BASE_SEED = 20261125
DUST = 'MineralDust'


@dataclass(frozen=True)
class Sky:
    eddy: float = .55
    exchange: float = .09
    rh: float = .85
    mix: float = 0.
    base_temp: float = -11.6
    cloud_base: float = 240.
    sounding: bool = False
    moisture_delta: float = 0.
    displacement: float = -1.
    period: float = 1800.
    delivered_flux: float = 0.
    background_ccn: float = 50.


BASE = Sky()
SKIES = {
    'reference': BASE,
    'weak_dry': Sky(.25, .035, .70, .5, -11.6, 360.),
    'vigorous_dry': Sky(.85, .14, .70, 4., -11.6, 240.),
    'moist_mixed': Sky(.55, .09, 1., 4., -11.6, 240.),
    'warm_thin': Sky(.35, .05, .85, 2., -8.6, 420.),
    'cold_deep': Sky(.65, .09, .90, 2., -14.6, 120.),
}
for key, field, value in [
    ('mix05','mix',.5),('mix2','mix',2.),('mix8','mix',8.),
    ('inlet70','rh',.70),('inlet100','rh',1.),('weak_exchange','exchange',.035),
    ('strong_exchange','exchange',.14),('slow_roll','eddy',.25),('fast_roll','eddy',.85),
    ('warmer','base_temp',-8.6),('colder','base_temp',-14.6),
    ('thinner','cloud_base',420.),('deeper','cloud_base',120.)]:
    SKIES[key] = replace(BASE, **{field:value})


@dataclass(frozen=True)
class Experiment:
    sky: str
    donor: str
    seed: int
    grid: int = 10
    slope: float = .5
    plate: int = 0
    share: float = 1.
    loading: float = 1.
    hours: float = 6.
    dt_factor: float = 1.
    density: int = 48
    resident: str = DUST
    law: str = 'fixed'
    reset: int = 0
    grain_diameter: float = 1.
    grain_number: float = 1000.
    site_shift: float = 0.
    abundance: str = 'equal'


def rate_spectra():
    g = geometry(read_inputs(PACKET, False)[0])
    rows = []
    for source in SAMPLES:
        subset = g[g['sample'] == source]
        temp = float(subset.target.iloc[0])
        for law, power in [('fixed', 1.), ('aging', .6)]:
            _, rates, prior, _, posterior = train(subset, power=power)
            night = subset[(subset.phase == 'night') & (subset.h > 0)]
            for plate, wells in night.groupby('plate'):
                weights = posterior[(plate - 1)*100 + wells.id - 1].copy()
                weights[wells['shift'] > 80] = prior
                fresh = weights.mean(axis=0)
                aged = fresh * np.exp(-rates * 2**power)
                aged /= aged.sum()
                for rate, f, a in zip(rates, fresh, aged):
                    rows.append((source, law, temp, plate, rate, f, a))
    return pd.DataFrame(rows, columns=['source','law','temperature','plate','rate','fresh','aged'])


def spectra_builder(frame):
    @lru_cache(None)
    def spectrum(source, column, slope, plate, law):
        group = frame[(frame.source == source) & (frame.law == law)]
        if plate: group = group[group.plate == plate]
        weights = group.groupby('rate')[column].mean()
        power = .6 if law == 'aging' else 1.
        weights.index = weights.index * np.exp(power*slope*(14 + group.temperature.iloc[0]))
        assert abs(weights.sum() - 1) < 1e-9
        return weights

    def compose(e):
        resident = spectrum(e.resident, 'aged', e.slope, 0, e.law)
        source = e.donor.removeprefix('aged_')
        if source in ('closed','inactive'):
            source = e.resident
        if source == 'dust_rural_mix':
            fresh = .5*spectrum(DUST,'fresh',e.slope,e.plate,e.law).add(
                spectrum('RuralContinental','fresh',e.slope,e.plate,e.law), fill_value=0.)
        else:
            fresh = spectrum(source, 'aged' if e.donor.startswith('aged_') else 'fresh', e.slope,e.plate,e.law)
        table = pd.concat([resident.rename('aged'),fresh.rename('fresh')],axis=1).fillna(0)
        table.index.name = 'rate'
        return table.reset_index().assign(law=e.law,plate=0)
    return compose


def key(e):
    return json.dumps(asdict(e),sort_keys=True)


def groups(skies, donors, seeds, **kw):
    for i in seeds:
        for sky in skies:
            for donor in donors:
                yield Experiment(sky,donor,BASE_SEED+i,**kw)


CONTROLS = ['aged_'+DUST,'closed']
ALL = [*CONTROLS,*SAMPLES]
SKY_CORE = list(SKIES)[:6]
ONE_FACTOR = list(SKIES)[6:]
for label,mix in [('isdac',2.),('isdac_no_mix',0.),('isdac_mix8',8.)]:
    SKIES[label]=replace(BASE,mix=mix,sounding=True,background_ccn=200.)
for label,increment in [('qbase',0.),('qplus025',.00025),('qplus050',.00050)]:
    SKIES[label]=replace(BASE,mix=2.,rh=.60,moisture_delta=increment)
SKIES['very_cold']=replace(BASE,base_temp=-20.6,mix=2.)
for label,mix in [('mpace',2.),('mpace_no_mix',0.),('mpace_mix8',8.)]:
    SKIES[label]=replace(BASE,mix=mix,sounding=True)
for label,start in [('synthetic','mix2'),('mpace','mpace'),('isdac','isdac'),('dry','vigorous_dry')]:
    for amp,period in [(0.,1800.),(200.,1800.),(400.,1800.),(200.,600.)]:
        SKIES[f'{label}_roll{int(amp)}_{int(period)}']=replace(SKIES[start],displacement=amp,period=period)
for label in ['isdac','mpace','reference','vigorous_dry']:
    for mm in [1.2,4.,4.125,7.2,8.2,12.,20.]:
        SKIES[label+'_we'+str(mm).replace('.','p')]=replace(SKIES[label],exchange=np.pi*mm/1000)
SKIES['isdac_ccn50']=replace(SKIES['isdac'],background_ccn=50.)
SKIES['mpace_ccn200']=replace(SKIES['mpace'],background_ccn=200.)
for share in [.05,.1,.25,.5,1.]:
    SKIES[f'mpace_flux{int(100*share)}']=replace(SKIES['mpace'],delivered_flux=share*107.7/2.5e6)



def design(include_extra_replicates=True):
    # Reference first, then cross all six atmospheric source labels with all six clouds.
    yield from groups(['reference'],[DUST,'aged_'+DUST],range(1),hours=.5,density=12)
    yield from groups(SKY_CORE,ALL,range(8))
    yield from groups(SKY_CORE,['aged_'+s for s in SAMPLES if s!=DUST],range(8))
    yield from groups(ONE_FACTOR,ALL,range(4))
    for slope in [0.,.25,.75]:
        yield from groups(['reference','warm_thin','cold_deep'],ALL,range(4),slope=slope)
    for plate in range(1,6):
        yield from groups(['reference','vigorous_dry'],[*SAMPLES,'aged_'+DUST],range(4),plate=plate)
    yield from groups(SKY_CORE,ALL,range(5),grid=20)
    for grid in [30,40]:
        yield from groups(['reference','vigorous_dry','warm_thin'],['aged_'+DUST,'closed',DUST,'RuralContinental','CombustionDominated','Marine'],range(2),grid=grid)
    for sky in SKY_CORE:
        for source in [DUST,'Marine','RuralContinental','CombustionDominated']:
            for share in [0.,.25,.5,.75]:
                yield from groups([sky],[source],range(4),share=share)
    for law,reset in [('aging',0),('aging',1)]:
        yield from groups(SKY_CORE,[DUST,'aged_'+DUST,'Marine','RuralContinental','closed'],range(6),law=law,reset=reset)
    for level in [.1,10.]:
        yield from groups(SKY_CORE,ALL,range(4),loading=level)
    for e in groups(SKY_CORE,['dust_rural_mix'],range(8)):
        yield e
    for sky in ['reference','vigorous_dry','warm_thin']:
        for factor in [.5,.25]:
            yield from groups([sky],ALL,range(3),dt_factor=factor)
    if not include_extra_replicates:
        return
    # Keep paired scientific comparisons coming as time permits.
    for i in range(2,1000):
        yield from groups(SKY_CORE,['aged_'+DUST,'closed',*SAMPLES], [i],grid=30)
        yield from groups(['reference','vigorous_dry','warm_thin'],['aged_'+DUST,DUST,'RuralContinental'],[i],grid=40)


def simulate(e, compose, out):
    s = SKIES[e.sky]
    dt = (10 if e.grid==10 else 5 if e.grid==20 else 2.5)*e.dt_factor
    c = cloud.Case(name=e.donor,n=e.grid,per=round(e.density*(e.grid/10)**2),
                   dt=dt,hours=e.hours,entrain=0. if e.donor=='closed' else s.exchange,
                   eddy=s.eddy,rh=s.rh,source='inert' if e.donor=='inactive' else 'fresh',
                   beta=e.slope,seed=e.seed,mixing=s.mix,loading=e.loading,
                   law=e.law,reset=e.reset,matched_closed=e.donor=='closed',active_share=e.share)
    diagnostics={}
    def endpoint(a):
        for tag,mask in [('incoming',a['birth']>=0),('resident',a['birth']<0)]:
            mask &= (a['carrier']==1)&(a['fate']!=0)
            wt=a['weight'][mask]; norm=1600*600*1000
            for label, values in [('ever_wet',a['wets']>0),('wet_h',a['wettime']/3600),
                                  ('dried',a['dries']>0),('fell_out',a['fate']==3),('exported',a['fate']==2)]:
                diagnostics[tag+'_'+label]=np.sum(wt*values[mask])/norm
    inlet_loading=None
    if e.abundance=='air_volume':
        amounts=loadings()
        incoming=e.donor.removeprefix('aged_')
        if incoming=='closed':incoming=e.resident
        aged=e.donor.startswith('aged_')
        inlet_loading=amounts.loc[incoming,'aged_per_litre' if aged else 'entry_per_litre']
        c=replace(c,loading=amounts.loc[e.resident,'aged_per_litre'])
    # For the aging challenger, incoming survivors carry their two-hour wet clock.
    material=e.donor.removeprefix('grain_') if e.donor.startswith('grain_') else None
    rates = compose(replace(e,donor='inactive') if material else e)
    treatment=Mineral(material,e.grain_diameter,e.grain_number,e.site_shift,e.seed) if material else None
    # Copy rates only onto residents. Mineral arrivals use permanent active sites.
    if material:
        c=replace(c,source='inert')
    inlet_density=None
    field=mpace if e.sky.startswith('mpace') else isdac
    if s.sounding:
        inlet_density=field.inlet()
    elif s.moisture_delta:
        top_c=s.base_temp-.006*(600-300/e.grid)
        inlet_density=s.rh*isdac.saturation(top_c+273.15)/(461.5*(top_c+273.15))+1.1*s.moisture_delta
    if e.law=='aging' and e.donor.startswith('aged_'):
        assert e.donor=='aged_'+e.resident and e.plate==0
        c = replace(c,source='aged')
    start=time.monotonic()
    h,m=cloud.run(c,rates,out,base_temperature=s.base_temp,cloud_base=s.cloud_base,
                  closed_entrain=s.exchange,endpoint=endpoint,treatment=treatment,
                  profile=field.column if s.sounding else None,inlet_density=inlet_density,
                  lower_restore=not s.sounding,background_ccn=s.background_ccn,
                  inlet_loading=inlet_loading,flow=Roll(s.displacement,s.period) if s.displacement>=0 else None,
                  lower_source=(supply,s.delivered_flux) if s.delivered_flux else None)
    diagnostics['mean_LWP']=float(np.trapezoid([r['LWP'] for r in h],[r['t_h'] for r in h])/e.hours)
    if inlet_loading is not None:
        diagnostics.update(inlet_carriers_l=inlet_loading,resident_carriers_l=c.loading)
    if treatment:
        diagnostics.update(max_site_probability=treatment.maximum,admitted_grains=treatment.grains_per_litre)
    if inlet_density is not None:diagnostics['inlet_vapor_kg_m3']=inlet_density
    return dict(experiment=asdict(e),sky=asdict(s),integrator=m,wall=time.monotonic()-start,
                values=h[-1],diagnostics=diagnostics,
                evolution=[{k:row[k] for k in ('t_h','primary','suspended','LWP','resident_primary','fresh_primary')}
                           for row in h if row['t_h'] in (0,1,2,4,6,8,12)])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--seconds',type=float,default=7200)
    ap.add_argument('--cases',type=Path);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(exist_ok=True,parents=True)
    clock=out/'budget.json'
    if not clock.exists():
        clock.write_text(json.dumps({'start':time.time(),'budget_seconds':args.seconds},indent=2))
    budget=json.loads(clock.read_text());deadline=budget['start']+budget['budget_seconds']
    rates=rate_spectra();rates.to_csv(out/'spectra.csv',index=False);compose=spectra_builder(rates)
    ledger=out/'runs.jsonl'
    done=set(); costs={}
    def learn(e,wall):
        group=(e.grid,e.dt_factor,e.hours,e.density,SKIES[e.sky].exchange,SKIES[e.sky].mix>0)
        costs[group]=max(wall,costs.get(group,0))
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            r=json.loads(line);e=Experiment(**r['experiment']);done.add(key(e));learn(e,r['wall'])
    supplied=args.cases is not None
    if supplied:
        text=gzip.decompress(args.cases.read_bytes()).decode() if args.cases.suffix=='.gz' else args.cases.read_text()
        queue=(Experiment(**x) for x in json.loads(text))
    else:
        queue=design()
    class TimeLimit(Exception):
        pass

    def stop_at_limit(signum, frame):
        raise TimeLimit

    signal.signal(signal.SIGALRM, stop_at_limit)
    def finish(partial=None):
        now=time.time()
        result={'completed':len(done),'end':now,'elapsed':now-budget['start']}
        if partial is not None:
            result['incomplete_setting']=asdict(partial)
        (out/'finished.json').write_text(json.dumps(result,indent=2))

    with ledger.open('a',buffering=1) as log:
        def execute(e):
            if (out/'pause_request').exists():
                print('Checkpoint requested between runs.',flush=True);raise SystemExit(0)
            remaining=deadline-time.time()
            if remaining<=0:
                finish()
                raise SystemExit(0)
            signal.setitimer(signal.ITIMER_REAL,remaining)
            try:
                result=simulate(e,compose,out)
            except TimeLimit:
                finish(e)
                raise SystemExit(0)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
            log.write(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n');log.flush()
            done.add(key(e));learn(e,result['wall'])
            if len(done)%100==0: print('completed',len(done),e,flush=True)
        stamp=None
        for e in queue:
            if key(e) in done:continue
            # Read newly planned challenge cases between runs. They use the same clock and checkpoints.
            priority_file=out/'priority_cases.json'
            if not supplied and priority_file.exists() and priority_file.stat().st_mtime_ns!=stamp:
                stamp=priority_file.stat().st_mtime_ns
                for item in json.loads(priority_file.read_text()):
                    ex=Experiment(**item)
                    group=(ex.grid,ex.dt_factor,ex.hours,ex.density,SKIES[ex.sky].exchange,SKIES[ex.sky].mix>0)
                    time_left=deadline-time.time()
                    if key(ex) not in done and time_left>max(60.,1.6*costs.get(group,0.)):
                        execute(ex)
            remaining=deadline-time.time()
            if remaining<2.:break
            estimate=costs.get((e.grid,e.dt_factor,e.hours,e.density,SKIES[e.sky].exchange,SKIES[e.sky].mix>0),
                               .9*(e.grid/10)**2*(10/(10 if e.grid==10 else 5 if e.grid==20 else 2.5))*e.hours/6/e.dt_factor)
            if remaining<max(2.,1.4*estimate):
                if supplied:break
                # Spend the tail on cheap complete pairs/realizations of the reference cloud.
                for i in range(1000,2000):
                    if deadline-time.time()<2.3:break
                    for b in [DUST,'aged_'+DUST]:
                        x=Experiment('reference',b,BASE_SEED+i)
                        if key(x) not in done and deadline-time.time()>1.5:execute(x)
                break
            if key(e) not in done:
                execute(e)
    finish()

if __name__=='__main__':
    main()
