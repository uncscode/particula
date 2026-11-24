"""Compact comparison of individual mechanisms, joint cases and the signed AIS constraint."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
out=Path(sys.argv[1])
df=pd.read_csv(out/'timeseries.csv')
signed=pd.read_csv(out/'charge-timeseries.csv').query("case=='charge'")
obs=pd.read_csv(Path(__file__).parent/'ion-reference.csv')
fig,(a,b,c)=plt.subplots(3,1,figsize=(10,8.6),sharex=True,layout='constrained',
                        gridspec_kw={'height_ratios':[1.5,1,1]})
a.plot(df.archive_h,df.observed_tail,color='#202020',lw=2.5,label='Measured')
for name,label,color,style in [('reference','Reference','#666666','--'),('charge','Charge tracking','#00678b','-'),
        ('both','Charge + evaporation, acid-rich','#c24f28','-'),
        ('both_paired_matched_capture','Extra ammonia + molecular capture','#338259','--'),
        ('both_chamber_fit','Charge + evaporation, earlier chamber fit','#805498','-.')]:
    if name in df:a.plot(df.archive_h,df[name],ls=style,lw=1.7,label=label,color=color)
for v in (1000,5000):a.axhline(v,color='gray',ls=':',lw=0.8)
a.set_yscale('symlog',linthresh=150);a.set_yticks([0,1000,5000,15000],['0','1,000','5,000','15,000'])
a.set_ylim(0,25000)
for v in (1000,5000):
    hit=df[df.observed_tail>=v].iloc[0]
    a.scatter(hit.archive_h,hit.observed_tail,s=25,color='black',zorder=6)
a.set_ylabel('9.7 nm and larger\nparticles/cm³')
a.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False,fontsize=9)
for panel,pol,color in [(b,'negative','#244699'),(c,'positive','#a44239')]:
    mask=obs.archive_h.between(df.archive_h.min(),df.archive_h.max())
    unique=obs[mask].drop_duplicates('ais_native_row')
    panel.scatter(unique.ais_native_h,unique[f'ais_{pol}_ge2p2_cm3'],s=8,color='#777777',alpha=.65,label='Measured')
    panel.plot(signed.archive_h,signed[f'{pol}_cm3'],lw=1.8,color=color,label='Charge tracking')
    panel.set_ylabel(f'{pol.capitalize()} particles\n≥2.2 nm, particles/cm³')
    panel.yaxis.set_major_locator(MaxNLocator(4))
    panel.legend(frameon=False,loc='upper left',ncol=2,fontsize=9)
for sid in [9,11,14]:
    z=obs[obs.stage==sid].archive_h
    for panel in (a,b,c):panel.axvspan(z.min(),z.max(),color='#96b5c2',alpha=.14,zorder=0)
c.set_xlabel('Archive time (hours)  ·  Shading indicates clearing field on')
c.set_xlim(df.archive_h.min(),df.archive_h.max())
fig.savefig(out/'charge-evaporation.png',dpi=150)
