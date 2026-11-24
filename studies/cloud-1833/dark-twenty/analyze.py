# %% Summarize the archived cases and plot their measured and predicted crossings.
from pathlib import Path
import sys
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(sys.argv[1]).resolve()
cycles=sorted(root.glob('[0-9][0-9]-*'))
assert len(cycles)==20
read=lambda f:json.loads(f.read_text())
summary=[];thresholds=[];by_stage=[];restarts=[]
for folder in cycles:
    spec=read(folder/'proposal.json')
    meta=read(folder/'execution.json')
    runs=pd.read_csv(folder/'runs.csv')
    cs=pd.read_csv(folder/'candidates.csv')
    w=meta['winner']
    assert len(runs)==meta['cases'] and np.isfinite(cs.cost).all()
    ref=runs.query('candidate==@w and role=="test" and window=="X"').iloc[0]
    summary.append(dict(cycle=folder.name,question=spec['question'],cases=len(runs),candidates=len(cs),
        seconds=meta['seconds'],cells=meta['cells'],chosen=w,config=cs.config.iloc[w],
        selection_cost=cs.cost.iloc[w],budget_residual=runs.budget_residual.max(),
        **{k:ref[k] for k in ['first_min','second_min','rise_min','retention','mode12']}))
    series=pd.read_csv(folder/'series.csv.gz')
    for (candidate,window), x in series.groupby(['candidate','window']):
        if window=='12':
            full=series.query('candidate==@candidate and window=="X" and stage==12')
            restarts.append(dict(cycle=folder.name,candidate=candidate,observed_start=x.observed.iloc[0],reset_start=x.predicted.iloc[0],continuous_start=full.predicted.iloc[0]))
        if window!='X':continue
        for threshold in (1000,5000):
            for samples in (1,3,5):
                timing=[]
                for col in ['observed','predicted']:
                    hits=(x[col].to_numpy()>=threshold)
                    hit=np.flatnonzero(np.convolve(hits,np.ones(samples,dtype=int),'valid')==samples)
                    timing.append(x.archive_h.iloc[hit[0]] if len(hit) else np.nan)
                thresholds.append(dict(cycle=folder.name,candidate=candidate,threshold=threshold,samples=samples,observed_h=timing[0],predicted_h=timing[1],error_min=60*(timing[1]-timing[0])))
        for stage,g in x.groupby('stage'):
            p=g[['p1','p2','p3','p4']].to_numpy();o=g[['o1','o2','o3','o4']].to_numpy()
            by_stage.append(dict(cycle=folder.name,candidate=candidate,stage=stage,band_loss=np.abs(np.log((p+25)/(o+25))).mean(),
                observed_final=g.observed.iloc[-1],predicted_final=g.predicted.iloc[-1]))
pd.DataFrame(summary).to_csv(root/'cycle-results.csv',index=False)
thr=pd.DataFrame(thresholds);thr.to_csv(root/'thresholds.csv',index=False)
pd.DataFrame(restarts).to_csv(root/'restart.csv',index=False)
st=pd.DataFrame(by_stage);st.to_csv(root/'stage-errors.csv',index=False)
# %% Resample paired, whole stages. For four stages enumerate all 256 draws.
rng=np.random.default_rng(1833)
unc=[]
def compare(group, ref, choices, draws, label):
    for other in choices:
        diff=group[other].to_numpy()-group[ref].to_numpy()
        interval=np.quantile(np.exp(diff[draws].mean(1)),[.025,.975])
        unc.append(dict(comparison=label,reference=ref,alternative=other,stages=len(group),error_factor_ratio=np.exp(diff.mean()),low=interval[0],high=interval[1],stage_wins=int((diff<0).sum())))
dark=pd.read_csv(cycles[6]/'runs.csv').query('role=="fit"').pivot(index='window',columns='candidate',values='loss')
compare(dark,5,[4,6,7,8,9],np.array(list(itertools.product(range(4),repeat=4))),'individual dark channels, cycle 07')
exc=st.query('cycle=="20-refined-transfer" and stage<18').pivot(index='stage',columns='candidate',values='band_loss')
compare(exc,0,[1,2,3,4,5],rng.integers(0,9,(10000,9)),'excimer stages 09–17, cycle 20')
pd.DataFrame(unc).to_csv(root/'stage-resampling.csv',index=False)
# %% Replot the two threshold crossings, including repeated crossings.
x=pd.read_csv(cycles[-1]/'series.csv.gz').query('window=="X"')
fig,axs=plt.subplots(1,2,figsize=(10.4,4.),layout='constrained')
colors=['#6a7080','#007c74','#b95a21'];labels=['Earlier wall rate','Dark fit, fixed size dependence','Dark fit, adjustable size dependence']
for ax,(a,b,low,high,cut) in zip(axs,[(8.12,8.91,0,2300,1000),(9.80,10.86,4000,6300,5000)]):
    ax.axhline(cut,c='#999999',lw=1,ls='--')
    for k,(color,label) in enumerate(zip(colors,labels)):
        g=x.query('candidate==@k and @a<=archive_h<=@b')
        if k==0:ax.plot(g.archive_h,g.observed,'o',ms=3,color='#222222',label='Measured, every 2 minutes')
        ax.plot(g.archive_h,g.predicted,lw=2,color=color,label=label)
    ax.set(xlabel='Archive time (hours)',ylim=(low,high),title=f'{cut:,} particles/cm³ crossing')
    ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Number above 9.7 nm (particles/cm³)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.savefig(root/'crossings.png',dpi=180)
