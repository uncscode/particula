"""Summarize inlet exposure, stage uncertainty and numerical sensitivity."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1])
flux=pd.read_csv(out/'boundary.csv')
cases=pd.read_csv(out/'cases.csv')
scores=pd.read_csv(out/'stages.csv')
tails=pd.read_csv(out/'trajectories.csv.gz')
time=flux.archive_h.to_numpy();dt=np.diff(time)*3600
records=[]
for nm in cases.inlet.unique():
    f=flux[nm].to_numpy()
    rates=f[:-1] if nm.startswith('prior_') else .5*(f[:-1]+f[1:])
    # Allocate halves of a stage-transition interval to each adjacent stage.
    for st in range(9,19):
        half=.5*((flux.stage.to_numpy()[:-1]==st).astype(float)+(flux.stage.to_numpy()[1:]==st))
        records.append(dict(inlet=nm,stage=st,integrated_cm3=np.sum(dt*rates*half),mean_cm3_s=np.average(rates,weights=dt*half)))
pd.DataFrame(records).to_csv(out/'supply-by-stage.csv',index=False)
draw=np.random.default_rng(18330509).integers(0,9,(10000,9))
rows=[]
for (inlet,res),df in scores.query('stage<18').groupby(['inlet','cells']):
    matrix=df.pivot(index='stage',columns='model',values='spectral_factor')
    for name in matrix:
        delta=np.log(matrix[name]/matrix['frozen']).to_numpy()
        lo,hi=np.exp(np.quantile(delta[draw].mean(1),[.025,.975]))
        rows.append(dict(cells=res,inlet=inlet,model=name,stage_error_factor_ratio=np.exp(delta.mean()),low=lo,high=hi,
                         stages_improved=sum(delta<0)))
pd.DataFrame(rows).to_csv(out/'stage-uncertainty.csv',index=False)
a=cases[cases.cells==192].set_index(['inlet','model']);b=cases[cases.cells==288].set_index(['inlet','model'])
diff=b[['first_error_min','second_error_min','rise_min','dark_retained','mode12_nm','cut_mapping_max_shift_min']]-a[['first_error_min','second_error_min','rise_min','dark_retained','mode12_nm','cut_mapping_max_shift_min']]
diff.to_csv(out/'refinement.csv')
fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained',width_ratios=[1.6,1])
ax,bx=axes[0];cx,dx=axes[1]
lo=flux[['direct_hard','direct_attractive','from4','from5p9','balance']].min(1);hi=flux[['direct_hard','direct_attractive','from4','from5p9','balance']].max(1)
ax.fill_between(time,lo,hi,color='#b3d9cc',alpha=.65,label='Measurement-based scenarios')
ax.plot(time,flux.direct,color='#00846f',label='Central estimate',lw=1.5)
ax.step(time,flux.prior_passages,where='post',color='#9e500f',label='Counted in previous model',lw=1.25)
ax.set(xlim=(7.2,11.3),ylabel='5 nm influx (particles/cm³/s)',xlabel='Archive time (hours)');ax.legend(frameon=False,fontsize=8)
def subset(inlet,model):return tails[(tails.inlet==inlet)&(tails.model==model)]
series=[('prior_passages','frozen','#9e500f','Previous influx'),('direct','frozen','#00846f','Central influx'),('direct','prior_wall','#516fb0','Central influx, lower wall loss')]
ref=subset('direct','frozen')
for panel in (cx,dx):
    x=ref[ref.stage==18] if panel is dx else ref
    panel.plot(x.archive_h,x.observed_tail/(x.observed_tail.iloc[0] if panel is dx else 1),color='#202426',lw=2.6,label='Measured')
    for inlet,model,color,label in series:
        z=subset(inlet,model)
        if panel is dx:z=z[z.stage==18]
        panel.plot(z.archive_h,z.predicted_tail/(z.predicted_tail.iloc[0] if panel is dx else 1),color=color,lw=1.6,label=label)
cx.set(xlim=(7.2,12.8),ylim=(0,11000),ylabel='9.7–64.7 nm particles/cm³',xlabel='Archive time (hours)')
cx.legend(frameon=False,fontsize=8)
for cut in (1000,5000):cx.axhline(cut,color='gray',ls=':',lw=.7)
dx.set(xlabel='Archive time (hours)',ylabel='Fraction remaining in the dark')
bx.axhline(0,ls=':',color='gray');bx.axvline(0,ls=':',color='gray')
other=cases.query("cells==288 and inlet in ['direct','from4','from5p9','balance','direct_hard','direct_attractive']")
bx.scatter(other.first_error_min,other.second_error_min,color='#a7b8b6',s=14,alpha=.7)
for inlet,model,color,label in series:
    r=b.loc[inlet,model];bx.scatter(r.first_error_min,r.second_error_min,s=52,color=color)
bx.scatter([0],[0],color='black',marker='*',s=65)
bx.set(xlabel='1,000/cm³ crossing error (min)',ylabel='5,000/cm³ crossing error (min)')
fig.savefig(out/'boundary-test.png',dpi=150)
