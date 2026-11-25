# %%
"""Read the checkpointed CUDA runs, compute paired contrasts and make figure tables."""
import argparse
from dataclasses import fields
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

from campaign import Experiment, SAMPLES, DUST, BASE_SEED, SKY_CORE, ONE_FACTOR
EXPERIMENT=[f.name for f in fields(Experiment)]
METRICS=['primary','unique','fresh_primary','resident_primary','suspended','LWP','IWP','condensed','evaporated',
         'fresh_recycled_primary','resident_recycled_primary','fresh_live_ice','resident_live_ice',
         'incoming_wet_h','resident_wet_h','incoming_ever_wet','fresh_inflow_or_initial',
         'resident_rewetted','fresh_rewetted','mean_LWP']


def load(path):
    if path.suffix=='.gz':
        stream=gzip.open(path,'rt')
    else:stream=path.open()
    rows=[]
    with stream:
        for line in stream:
            if not line.endswith('\n'):continue
            x=json.loads(line)
            config=vars(Experiment(**x['experiment']))
            rows.append({**config,**x['values'],**x['diagnostics'],
                         'stage_wall':x['wall'],'integration_s':x['integrator']['seconds'],
                         'max_budget_error':x['integrator']['max_water_error'],
                         'records':x['integrator']['records'],'steps':x['integrator']['steps']})
    return pd.DataFrame(rows)


def paired(a,b,metric, prefer_independent=True):
    # Multiple seeds are the unit of uncertainty; all other varied parameters
    # must already have been fixed by the caller.
    a=a[['seed',metric]].dropna().set_index('seed')
    b=b[['seed',metric]].dropna().set_index('seed')
    assert a.index.is_unique and b.index.is_unique,metric
    z=a.join(b,how='inner',lsuffix='_a',rsuffix='_b')
    independent=z.index>=BASE_SEED+10000019
    if prefer_independent and independent.any():
        z=z[independent]
    n=len(z)
    if not n:return None
    d=z.iloc[:,0]-z.iloc[:,1]
    half=t.ppf(.975,n-1)*d.sem() if n>1 else np.nan
    return dict(n=n,spaced_seeds=bool((z.index>=BASE_SEED+10000019).all()),value=z.iloc[:,0].mean(),control=z.iloc[:,1].mean(),
                delta=d.mean(),lo=d.mean()-half,hi=d.mean()+half)


def select(d,**changes):
    # Match every non-varying configuration field, including numerical and source knobs.
    default=vars(Experiment('reference',DUST,BASE_SEED))
    conditions={**default,**changes}
    m=np.ones(len(d),bool)
    for name,value in conditions.items():
        if value is not None:m &= d[name].to_numpy()==value
    return d[m]


def lookup(d,**kw):
    return select(d,seed=None,**kw)


def tables(path,d):
    # Roundtrip copies of nominal cases make no additional independent draws.
    d=d.drop_duplicates(EXPERIMENT).copy()
    summary=[]
    def add(label,left,right,**labels):
        for metric in METRICS:
            x=paired(left,right,metric)
            if x:summary.append(dict(comparison=label,metric=metric,**labels,**x))
    skies=sorted(set(d.sky))
    for abundance in ['equal','air_volume']:
      for grid in [10,20,30,40,60,80]:
        for sky in skies:
            g=lookup(d,sky=sky,grid=grid,donor=None,abundance=abundance)
            if g.empty:continue
            donors=dict(tuple(g.groupby('donor')))
            for source in SAMPLES:
                if source not in donors:continue
                for label,ref in [('closed','closed'),('old_dust','aged_'+DUST),('own_survivors','aged_'+source)]:
                    if ref in donors:add(label,donors[source],donors[ref],sky=sky,source=source,grid=grid,abundance=abundance)
            # Mineral experiment uses a different carrier loading and nucleation law.
            for mineral in ['grain_quartz','grain_feldspar']:
                if mineral in donors and 'grain_inert' in donors:
                    add('inert_grains',donors[mineral],donors['grain_inert'],sky=sky,source=mineral,grid=grid,abundance=abundance)
    compact=pd.DataFrame(summary)
    compact.to_csv(path/'paired.csv',index=False)
    return compact


def panel(d,sky,source,**kw):
    args=dict(sky=sky,donor=source,**kw)
    g=lookup(d,**args)
    b=lookup(d,**{**args,'donor':'closed'})
    a=lookup(d,**{**args,'donor':'aged_'+DUST})
    return [paired(g,z,'primary') for z in [b,a]]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('ledger',type=Path)
    parser.add_argument('out',type=Path)
    a=parser.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    data=load(a.ledger)
    frame=tables(a.out,data)
    print('integrations',len(data),'unique',len(data.drop_duplicates(EXPERIMENT)),
          'integration seconds',round(data.integration_s.sum(),1),'largest water error',data.max_budget_error.max())
    print(frame.query("metric=='primary' and comparison=='closed' and grid==10 and abundance=='equal'")
          .pivot(index='sky',columns='source',values='delta').round(3).to_string())
