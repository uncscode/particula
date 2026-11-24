# %% Results and whole-stage uncertainty from the frozen comparisons
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(sys.argv[1]).resolve()
cycles=sorted(ROOT.glob('[0-9][0-9]-*'))
assert len(cycles)==20
diagnostics=[]; age_curves=[]; statistics=[]; details=[]
for folder in cycles:
    cfg=pd.read_csv(folder/'candidates.csv')
    meta=json.loads((folder/'execution.json').read_text())
    sol=pd.read_csv(folder/'runs.csv')
    assert len(sol)==meta['cases']
    comparisons=sol.query('window=="X" and role=="test"').set_index('candidate')
    for k,entry in cfg.iterrows():
        details.append(dict(cycle=folder.name,**entry.to_dict(),**comparisons.loc[k].to_dict()))
    statistics.append(dict(cycle=folder.name,cases=meta['cases'],candidates=len(cfg),
        winner=meta['winner'],cells=meta['cells'],seconds=meta['seconds'],
        max_number_residual=sol.budget_residual.max(),max_tag_residual=sol.tag_error.max()))
    df=pd.read_csv(folder/'series.csv.gz')
    for k,frame in df.query('window=="X"').groupby('candidate'):
        c=json.loads(cfg.config.iloc[k]);h=frame.archive_h.to_numpy()
        tr=frame.query('stage==12').archive_h.min();row=dict(cycle=folder.name,candidate=k)
        for v in [1000,5000]:
            for ns in (1,3):
                times=[]
                for column in ('observed','predicted'):
                    hit=np.convolve((frame[column].to_numpy()>=v).astype(int),np.ones(ns),mode='valid')
                    idx=np.flatnonzero(hit==ns)
                    times.append(h[idx[0]] if len(idx) else np.nan)
                row[f'error_{v}_{ns}_min']=(times[1]-times[0])*60
        for size in ('small','middle','tail'):
            row[size+'_at_first_crossing']=np.interp(10.15028,h,frame['model_'+size])
            row['observed_'+size]=np.interp(10.15028,h,frame['obs_'+size])
        for stage in (11,12,18):
            group=frame[frame.stage==stage]
            row[f'stage{stage}_tail_mae']=(group.predicted-group.observed).abs().mean()
        diagnostics.append(row)
        if 'offset' in c:
            for minute in range(0,91,2):
                hh=tr+(minute+c['offset'])/60
                age_curves.append(dict(cycle=folder.name,candidate=k,age_min=minute,**{
                    key:np.interp(hh,h,frame['tag_'+key]) for key in ('small','middle','tail')}))
pd.DataFrame(statistics).to_csv(ROOT/'execution-summary.csv',index=False)
pd.DataFrame(details).to_csv(ROOT/'case-summary.csv',index=False)
dg=pd.DataFrame(diagnostics);dg.to_csv(ROOT/'crossing-checks.csv',index=False)
ag=pd.DataFrame(age_curves);ag.to_csv(ROOT/'cohort-ages.csv.gz',index=False,compression={'method':'gzip','mtime':0})
# %% Enumerate paired draws of complete dark intervals over a common broad rate grid.
fold=cycles[2];fits=pd.read_csv(fold/'runs.csv').query('role=="fit"').pivot(index='window',columns='candidate',values='loss')
rates=np.array([json.loads(s)['eddy'] for s in pd.read_csv(fold/'candidates.csv').config])
per_stage={stage:rates[np.argmin(vec)] for stage,vec in fits.iterrows()}
draws=np.array(list(itertools.product(range(4),repeat=4)))
best=rates[fits.to_numpy()[draws].mean(1).argmin(axis=1)]
hist=pd.Series(best).value_counts().sort_index()
unc={'stages':per_stage,'rate_grid_s-1':list(rates),'resampled_rates_and_frequencies':hist.to_dict(),
     'percentiles_2p5_50_97p5':np.quantile(best,[.025,.5,.975]).tolist(),'draws':len(draws)}
(ROOT/'stage-uncertainty.json').write_text(json.dumps(unc,indent=2)+'\n')
# %% Dark constraints and same-age pulse dynamics
fig,ax=plt.subplots(1,2,figsize=(10.1,4.1),layout='constrained')
for i,clr,lab in [(2,'#006f75','9.9–63.5 nm'),(4,'#a85421','7.7–9.52 nm')]:
    score=pd.read_csv(cycles[i]/'candidates.csv')
    rate=[json.loads(s)['eddy'] for s in score.config]
    ax[0].plot(rate,100*np.expm1(score.cost-score.cost.min()),'o-',color=clr,label=lab)
ax[0].set(xscale='log',xlabel='Near-wall eddy parameter (s⁻¹)',ylabel='Increase in error factor from best fit (%)',title='Dark spectra favor different rates')
for k,clr,lab in [(1,'#006f75','20 min before transition'),(2,'#a85421','20 min after transition')]:
    g=ag.query('cycle=="17-equal-age-acid" and candidate==@k')
    ax[1].plot(g.age_min,g['tail'],color=clr,lw=2,label=lab)
ax[1].set(xlabel='Minutes since pulse center',ylabel='Tagged lineages above 9.7 nm (cm⁻³)',title='Arrival compared at the same age')
for a in ax:a.legend(frameon=False,fontsize=8);a.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'wall-and-arrival.png',dpi=180)
