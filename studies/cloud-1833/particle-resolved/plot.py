"""Replot retained ensemble concentrations, birth-cohort survival and timing."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root=Path(sys.argv[1])
ts=pd.read_csv(root/'timeseries.csv')
co=pd.read_csv(root/'cohorts.csv').query('case=="discrete" and pathway=="all"').set_index('stage')
fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
a,b=axes
cut=ts.archive_h.between(7.5,12.6)
a.axvspan(7.816944,9.816944,color='#efebdf',label='Stage 11')
for key,label,color,linestyle in [('observed_tail','Observed','#222222','-'),('discrete','Particle-resolved baseline','#16736a','-'),
        ('half birth acid response in stage 11','Half the acid increase for births','#b65628','--'),
        ('half growth acid response in stage 11','Half the acid increase for cluster growth','#62539d','-.')]:
    a.plot(ts.archive_h[cut],ts.loc[cut,key],color=color,ls=linestyle,lw=1.9,label=label)
for v in (1000,5000):a.axhline(v,color='0.72',lw=.7,zorder=0)
a.set(xlabel='Archive time (h)',ylabel='9.7–64.7 nm particles (cm⁻³)',title='Population growth through the medium-light pulse')
a.legend(frameon=False,fontsize=8,loc='upper left')
stages=[10,11,14];pos=np.arange(3);h=co.loc[stages]
b.errorbar(pos,h.median_min,yerr=[h.median_min-h.p10_min,h.p90_min-h.median_min],fmt='o',color='#8a5526',capsize=4)
for x,(_,r) in enumerate(h.iterrows()):
    b.annotate(f'{r.fraction_reaching:.0%} survive',(x,r.p90_min),textcoords='offset points',xytext=(0,7),
               ha='center',fontsize=9)
b.set(xticks=pos,xticklabels=['10\nLow','11\nMedium','14\nHigh'],xlim=(-.48,2.48),ylim=(0,26),
      xlabel='Birth stage and light level',ylabel='Transit from 0.8 to 2.9 nm (min)',title='Median and 10th–90th percentiles\namong surviving clusters')
fig.savefig(root/'particle-results.png',dpi=165)
