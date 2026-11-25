# %%
"""Replot the Lund study solely from its saved numerical results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from particula.plotting.toc import add_toc_logo
from study import SAMPLES

NAMES=['Marine','Mineral dust','Continental pristine','Continental polluted','Combustion','Rural continental']
TEAL='#007C78';RUST='#A34821';INK='#263340';BLUE='#47739B';GREY='#666563';PURPLE='#796388'
COLORS={'population':GREY,'persistent':RUST,'aging':TEAL,'redrawn':BLUE}
LABELS={'population':'Population clock','persistent':'Fixed well rate','aging':'Rate declines during hold','redrawn':'Rate varies by cycle'}


def render(out):
    q=pd.read_csv(out/'intervals.csv').query("scope=='all'").set_index('metric')
    with plt.style.context('particula.toc'):
        fig,ax=plt.subplots(figsize=(3,3));fig.subplots_adjust(left=.19,right=.95,bottom=.19,top=.71)
        fig.text(.10,.89,'Lund droplet arrays',weight='bold',fontsize=11)
        fig.text(.10,.80,'Measured fraction freezing (%)',fontsize=9)
        for j,(label,color,hat) in enumerate([('high',TEAL,''),('low',RUST,'///')]):
            ys=[q.loc[p+'_'+label,'mean'] for p in ['early','late']]
            xs=np.arange(2)+(.18 if j==0 else -.18)
            ax.bar(xs,ys,width=.31,color=color,hatch=hat,edgecolor=INK,lw=.5,label=label+' risk')
            for x,y in zip(xs,ys):
                ax.text(x,y+1.4,f'{y:.0f}%',ha='center',fontsize=8)
        ax.set(ylim=(0,76),xticks=[0,1],xticklabels=['0–30 min','2–10 h'],yticks=[0,25,50,75])
        ax.spines[['top','right']].set_visible(False);ax.grid(False)
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.52,.015),ncol=2,frameon=False,fontsize=8,handlelength=1)
        add_toc_logo(fig,corner='top right');fig.savefig(out/'toc.png',dpi=300);plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    ref=pd.read_csv(out/'paper_reference.csv')
    summaries = out / 'well-means.csv'
    d = pd.read_csv(summaries if summaries.exists() else out / 'well_forecasts.csv.gz')
    fig,axes=plt.subplots(2,3,figsize=(10.4,5.8),sharex=True,sharey=True,layout='constrained')
    for s,title,ax in zip(SAMPLES,NAMES,axes.flat):
        g=ref[ref['sample']==s].sort_values('h')
        measured=d[d['sample']==s]
        entry = measured.entry.iloc[0] if summaries.exists() else len(measured)
        baseline = 1 - entry / 500
        for model in ['population','persistent']:
            hours=[0,.5,2,4,6,8,10]
            ys=[baseline+(1-baseline)*measured[f'{model}_{t}'].mean() for t in hours]
            ax.plot(hours,100*np.array(ys),linestyle=':' if model=='population' else '--',
                    color=COLORS[model],label=LABELS[model])
        ax.plot(g.h,100*g.paper,color=BLUE,label='Published fit')
        ax.step(g.h,100*g.overnight,where='post',color=INK,lw=1.6,label='Overnight records')
        points=g.dropna(subset=['table'])
        ax.scatter(points.h,100*points.table,facecolor='white',edgecolor=BLUE,marker='s',zorder=5,label='Published Table 3')
        ax.set(title=title,xticks=[0,2,6,10],ylim=(0,100));ax.grid(axis='y',alpha=.15)
    fig.supxlabel('Time at the hold temperature (h)');fig.supylabel('Frozen droplets per 100 original droplets')
    axes[1,2].legend(loc='upper left',fontsize=7,handlelength=2,frameon=False)
    fig.savefig(out/'reference.png',dpi=175);plt.close(fig)
    fig,(a,b)=plt.subplots(1,2,figsize=(10.8,4.8),width_ratios=[.85,1.45],layout='constrained')
    for n,p in enumerate(['early','late']):
        for lab,mark,color,offset in [('low','o',RUST,-.10),('high','s',TEAL,.10)]:
            v=q.loc[p+'_'+lab];a.errorbar(n+offset,v['mean'],yerr=[[v['mean']-v.low],[v.high-v['mean']]],
                   marker=mark,ms=8,c=color,capsize=4,label=lab.title()+' predicted risk' if n==0 else None)
    a.set(xticks=[0,1],xticklabels=['First 30 min','Hours 2–10'],xlim=(-.5,1.5),ylim=(0,73),
          ylabel='Measured freezing within quartile (%)',title='Which droplets freeze?')
    a.legend(frameon=False,fontsize=8);a.grid(axis='y',alpha=.2)
    sc=pd.read_csv(out/'plate_scores.csv').query("variant=='main' and subset=='all'")
    box=pd.read_csv(out/'boxes.csv').query("reset==True and fraction==0")
    for i,s in enumerate(SAMPLES):
        obs=sc[(sc['sample']==s)&(sc.model=='persistent')].late
        b.scatter(obs,i+np.linspace(-.11,.11,len(obs)),s=17,marker='|',c=INK,alpha=.6,label='Measured plates' if i==0 else None)
        b.scatter(obs.mean(),i,marker='o',c=INK,s=32,label='Measured mean' if i==0 else None)
        for j,(model,marker) in enumerate([('population','+'),('persistent','x'),('aging','s'),('redrawn','^')]):
            g=box[(box['sample']==s)&(box.model==model)]
            x=100*g.late_ice_per_l.mean()
            b.scatter(x,i+(j-1.5)*.083,marker=marker,s=34,color=COLORS[model],label=LABELS[model] if i==0 else None)
    b.set(yticks=np.arange(6),yticklabels=NAMES,title='How much ice forms during hours 2–10?',
          xlabel='New ice per 100 droplets liquid at hold entry');b.invert_yaxis();b.grid(axis='x',alpha=.15)
    b.legend(loc='upper right',frameon=False,fontsize=7);fig.savefig(out/'forecasts.png',dpi=180);plt.close(fig)
    fig,(a,b)=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    box=pd.read_csv(out/'boxes.csv').query('reset==True')
    for model,mark in [('population','+'),('persistent','x'),('aging','s'),('redrawn','^')]:
        g=box[box.model==model].groupby('fraction').late_ice_per_l.mean()
        a.plot(100*g.index,g,marker=mark,label=LABELS[model],c=COLORS[model])
    ren=pd.read_csv(out/'renewal_observations.csv')
    f=np.linspace(0,1,5)
    a.plot(100*f,ren.closed.mean()+f*ren.gain.mean(),c=INK,ls='--',label='From measured cohorts')
    a.set(xlabel='Surviving liquid exchanged at 2 h (%)',ylabel='Predicted new primary ice, hours 2–10 (L⁻¹)',
          title='Fresh material from the same plate');a.legend(fontsize=8,frameon=False);a.grid(alpha=.18)
    sx=pd.read_csv(out/'source_exchange.csv')
    for sample,mark in [('MineralDust','o'),('ContinentalPolluted','s')]:
        g=sx[(sx.source==sample)&(sx.resident==sample)]
        b.errorbar(100*g.active_share,g.gain,yerr=[g.gain-g.low,g.high-g.gain],marker=mark,capsize=3,label=NAMES[SAMPLES.index(sample)])
    b.axhline(0,c=INK,lw=.8);b.set(ylabel='Full exchange minus closed (L⁻¹)',xlabel='Incoming carriers with extract-like activity (%)',
          title='Dilution can reverse the response');b.legend(fontsize=8,frameon=False);b.grid(alpha=.18)
    fig.savefig(out/'replacement.png',dpi=175);plt.close(fig)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('output',type=Path);render(a.parse_args().output)
