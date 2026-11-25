# %%
"""Array-level uncertainty, timing, conditional ranking and renewal diagnostics."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from study import SAMPLES, geometry, read_inputs

SEED=251125


def summarize(d, cols, scope):
    rng=np.random.default_rng(SEED)
    means=[]
    for _,g in d.groupby('sample',sort=True):
        x=g[cols].to_numpy()
        draws=rng.integers(len(x),size=(4000,len(x)))
        means.append(x[draws].mean(axis=1))
    distribution=np.mean(means,axis=0)
    lows,highs=np.quantile(distribution,[.025,.975],axis=0)
    centers=d.groupby('sample')[cols].mean().mean()
    return [dict(scope=scope,metric=c,mean=centers[c],low=l,high=h) for c,l,h in zip(cols,lows,highs,strict=True)]


def challenge(pkt,out):
    d=pd.read_csv(out/'well_forecasts.csv.gz')
    raw,_,_=read_inputs(pkt,False); raw=geometry(raw)
    ramp=(raw[(raw.phase=='ramp')&(raw['shift']<=80)]
          .groupby(['sample','plate','id']).temp.median().rename('ramp'))
    d=d.join(ramp,on=['sample','plate','id'])
    maxima=(d[d.h<=10].groupby(['sample','plate','seconds']).size()
              .groupby(level=[0,1]).max().rename('largest_group'))
    d=d.join(maxima,on=['sample','plate']).fillna({'largest_group':0})
    s=pd.read_csv(out/'plate_scores.csv')
    rows=[]; quantiles=[]; renewable=[]; pvalues=[]
    for (name,plate),g in d.groupby(['sample','plate']):
        f=lambda t: (g.h<=t).mean()
        base=f(10)-f(2)
        # Pulse replacement at 2h of surviving liquid, new arrivals drawn from
        # the same entry-liquid distribution. With no pulse there is zero import.
        gain=(1-f(2))*f(8)-base
        renewable.append(dict(sample=name,plate=plate,f2=f(2),f8=f(8),f10=f(10),
                              closed=base,gain=gain,largest_group=g.largest_group.iloc[0]))
        one=s[(s['sample']==name)&(s.plate==plate)&(s.variant=='main')&(s.subset=='all')].set_index('model')
        r=dict(sample=name,plate=plate,largest_group=g.largest_group.iloc[0],
               measured=one.late.iloc[0],population=one.loc['population','predicted'],
               persistent=one.loc['persistent','predicted'],
               abs_error_gain=one.loc['population','abs_error']-one.loc['persistent','abs_error'],
               bias=one.loc['persistent','error'],pop_bias=one.loc['population','error'],
               id_gain_early=one.loc['persistent','id_gain_early'],id_gain_late=one.loc['persistent','id_gain_late'],
               ramp_gain_early=one.loc['ramp_only','brier_early']-one.loc['persistent','brier_early'],
               ramp_gain_late=one.loc['ramp_only','brier_late']-one.loc['persistent','brier_late'],
               late_given_survival=one.loc['persistent','conditional_late'],
               observed_given_survival=one.loc['persistent','observed_conditional'])
        for start,end,label in [(0,.5,'early'),(2,10,'late')]:
            a=g[g.h>start].copy()
            p0=a[f'persistent_{start}']
            a['risk']=(a[f'persistent_{end:g}']-p0)/(1-p0)
            a['event']=(a.h<=end).astype(int)
            # Use full plate denominator and retain ties by deterministic ID order.
            a=a.sort_values(['risk','id'])
            k=len(a)//4
            for group,b in [('low',a.iloc[:k]),('high',a.iloc[-k:])]:
                val=100*b.event.mean()
                quantiles.append(dict(sample=name,plate=plate,period=label,group=group,n=k,
                                      events=int(b.event.sum()),observed=val,expected=100*b.risk.mean()))
                r[f'{label}_{group}']=val
            r[f'{label}_spread']=r[f'{label}_high']-r[f'{label}_low']
        rows.append(r)
    rows=pd.DataFrame(rows); q=pd.DataFrame(quantiles);ren=pd.DataFrame(renewable)
    rows.to_csv(out/'paired_metrics.csv',index=False);q.to_csv(out/'rank_quartiles.csv',index=False)
    ren.to_csv(out/'renewal_observations.csv',index=False)
    ci=[]
    metrics=list(rows.columns[3:])
    for scope,g in [('all',rows),('without_tied_bursts',rows[rows.largest_group<5])]:
        ci.extend(summarize(g,metrics,scope))
    for scope,g in [('renewal',ren),('renewal_without_bursts',ren[ren.largest_group<5])]:
        ci.extend(summarize(g,['gain','closed'],scope))
    pd.DataFrame(ci).to_csv(out/'intervals.csv',index=False)
    # Randomization conditional on plate, half-K ramp band, stage radius half.
    # Keep rank scores and event totals fixed. Use only safely mapped wells.
    rng=np.random.default_rng(SEED)
    for start,end,label in [(0,.5,'early'),(2,10,'late')]:
        a=d[(d.h>start)&(d['shift']<=80)&d.ramp.notna()].copy()
        a['risk']=(a[f'persistent_{end:g}']-a[f'persistent_{start}'])/(1-a[f'persistent_{start}'])
        a['event']=(a.h<=end).astype(int)
        a['band']=np.floor(2*a.ramp)
        cross=0.; null=np.zeros(4000); informative=0
        for _,g in a.groupby(['sample','plate','band','inner']):
            risk=g.risk.to_numpy()-g.risk.mean()
            event=g.event.to_numpy()
            if event.min()==event.max() or np.ptp(risk)==0:
                continue
            informative+=len(g)
            cross+=risk @ event
            shuffle=rng.random((4000,len(g))).argsort(axis=1)
            null+=(event[shuffle]*risk).sum(axis=1)
        low,high=np.quantile(null,[.025,.975])
        pvalues.append(dict(period=label,n=len(a),informative=informative,stat=cross,
                            null_low=low,null_high=high,p=(1+(null>=cross).sum())/4001))
    pd.DataFrame(pvalues).to_csv(out/'blocked_permutation.csv',index=False)
    print(pd.DataFrame(ci).query("scope=='all' or scope=='renewal'").round(4).to_string(index=False))
    print(pd.DataFrame(pvalues))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path); parser.add_argument('output',type=Path)
    a=parser.parse_args(); challenge(a.input,a.output)
