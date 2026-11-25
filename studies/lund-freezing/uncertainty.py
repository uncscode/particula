# %%
"""Resample whole short experiments and evaluate source-specific replacement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from study import SAMPLES,geometry,read_inputs,train,weibull

DATA=None


def initialize(pkt):
    global DATA
    d=geometry(read_inputs(pkt,False)[0]);DATA=dict(list(d.groupby('sample')))


def refit(task):
    name,index=task
    rng=np.random.default_rng([251125,SAMPLES.index(name),index])
    sample=DATA[name]
    short=sample[sample.phase=='short']
    draws=[]
    for _,p in short.groupby('plate'):
        runs=dict(list(p.groupby('run')))
        draws.extend(runs[r] for r in rng.choice(list(runs),len(runs),replace=True))
    fit=pd.concat([sample[sample.phase=='ramp'],*draws],ignore_index=True)
    ref,rates,prior,_,post=train(fit)
    vals=[]
    for plate,g in sample[(sample.phase=='night')&(sample.h>0)].groupby('plate'):
        w=post[((plate-1)*100+g.id-1).to_numpy()].copy();w[g['shift']>80]=prior
        t=g.h.to_numpy(); early=w @ (-np.expm1(-.5*rates))
        late=100*(w.mean(axis=0) @ (np.exp(-2*rates)-np.exp(-10*rates)))
        actual=100*np.mean((t>2)&(t<=10));pop=100*(weibull(10,ref)-weibull(2,ref))
        vals.append((name,index,plate,pop,late,actual,abs(pop-actual)-abs(late-actual),
                     np.mean((early.mean()-(t<=.5))**2-(early-(t<=.5))**2)))
    return vals


def resampling(pkt,out,n):
    with ProcessPoolExecutor(4,initializer=initialize,initargs=(pkt,)) as pool:
        rows=[r for v in pool.map(refit,[(s,i) for i in range(n) for s in SAMPLES],chunksize=3) for r in v]
    cols=['sample','draw','plate','population','persistent','measured','absolute_error_gain','early_brier_gain']
    df=pd.DataFrame(rows,columns=cols)
    df.to_csv(out/'short_refits.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    rng=np.random.default_rng(251126)
    outrows=[]
    for draw,g in df.groupby('draw'):
        for mode in ['fixed_plates','resampled_plates']:
            arr=g.sort_values(['sample','plate'])[cols[3:]].to_numpy().reshape(6,5,-1)
            if mode=='resampled_plates':
                arr=arr[np.arange(6)[:,None],rng.integers(0,5,(6,5))]
            p,i,o,ae,brier=arr.mean(axis=(0,1))
            outrows.append(dict(draw=draw,mode=mode,population=p,persistent=i,observed=o,
                                pop_bias=p-o,persistent_bias=i-o,abs_error_gain=ae,early_gain=brier))
    pd.DataFrame(outrows).to_csv(out/'refit_summary.csv',index=False)


def import_sources(out):
    r=pd.read_csv(out/'renewal_observations.csv')
    rng=np.random.default_rng(251127)
    values={};bs={}
    for s,g in r.groupby('sample'):
        x=g[['f2','f8','closed']].to_numpy();values[s]=x.mean(axis=0)
        bs[s]=x[rng.integers(0,len(x),size=(4000,len(x)))].mean(axis=1)
    rows=[]
    for source in SAMPLES:
        for dest in SAMPLES:
            if (source=='ContinentalPolluted')!=(dest=='ContinentalPolluted'):
                continue
            for f in [.25,.5,1]:
                # Two independently drawn inventory populations, exchange all surviving
                # liquid. f is the fraction of incoming carriers with donor-like activity.
                s2=1-values[dest][0]; closed=values[dest][2]; fresh=values[source][1]
                delta=s2*f*fresh-closed
                bootstrap=(1-bs[dest][:,0])*f*bs[source][:,1]-bs[dest][:,2]
                lo,hi=np.quantile(bootstrap,[.025,.975])
                rows.append(dict(source=source,resident=dest,active_share=f,survivor_threshold=closed/s2,
                                 fresh_probability=f*fresh,gain=delta,low=lo,high=hi))
    pd.DataFrame(rows).to_csv(out/'source_exchange.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--draws',type=int,default=240)
    a=parser.parse_args();import_sources(a.output);resampling(a.input,a.output,a.draws)
