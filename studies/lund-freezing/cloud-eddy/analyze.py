# %%
"""Compute paired contrasts and plot saved GPU fields and particle tracks."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

STAGES=('main','checks','plates','source','refine','environment','ventilation','fine_controls')
METRICS=('primary','unique','suspended','resident_primary','fresh_primary','resident_recycled_primary',
         'fresh_recycled_primary','LWP','IWP','condensed','evaporated','fallout_mass')
TEAL='#007e78';BLUE='#246799';RUST='#ae502a';INK='#263543'


def compare(group_a,group_b,label):
    a=group_a.set_index('seed');b=group_b.set_index('seed')
    seeds=a.index.intersection(b.index)
    assert len(seeds)>0 and a.index.is_unique and b.index.is_unique
    rows=[]
    for k in METRICS:
        x=a.loc[seeds,k]-b.loc[seeds,k];mean=x.mean()
        half=t.ppf(.975,len(x)-1)*x.sem() if len(x)>1 else np.nan
        rows.append(dict(comparison=label,metric=k,n=len(x),a=a.loc[seeds,k].mean(),b=b.loc[seeds,k].mean(),
                         difference=mean,low=mean-half,high=mean+half))
    return rows


def write_results(p):
    all_cases=pd.concat([pd.read_csv(p/(k+'_history.csv')).assign(stage=k) for k in STAGES])
    end=all_cases[all_cases.t_h==6];groups=dict(list(end.groupby('case')))
    comparisons=[('fresh',k) for k in ('closed_matched','aged','no_freezing','saturated_inlet')]
    comparisons += [(k,'aged') for k in ('inactive','quarter','third','half')]
    comparisons += [(k,k[:-5]+'aged') for k in groups if k.endswith('_fresh')]
    comparisons += [(n+'_fresh',f'closed_{m}') for n,m in [('twenty',20),('forty',40)]]
    rows=[]
    for a,b in comparisons:
        rows.extend(compare(groups[a],groups[b],a+' vs '+b))
    pd.DataFrame(rows).to_csv(p/'contrasts.csv',index=False)
    end.groupby('case')[list(METRICS)].mean().to_csv(p/'endpoints.csv')
    return all_cases


def plot(p,d):
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,
                         'axes.titleweight':'bold','axes.labelcolor':INK,'text.color':INK})
    fig,ax=plt.subplots(1,3,figsize=(13,4.0),layout='constrained')
    labels={'closed_matched':('Closed top',INK,'--'),'aged':('Survivor-like inflow',RUST,':'),
            'fresh':('Fresh-source inflow',TEAL,'-')}
    for name,(lab,col,ls) in labels.items():
        g=d[d.case==name].groupby('t_h')
        for a,k in zip(ax[:2],['primary','LWP']):
            m=g[k].mean()
            a.plot(m.index,m,lw=2.4,c=col,ls=ls,label=lab)
            a.fill_between(m.index,m-g[k].sem()*2.365,m+g[k].sem()*2.365,color=col,alpha=.10)
    for a,title,ylab in zip(ax[:2],['Ice formed over six hours','Liquid retained in the layer'],
                            ['Cumulative primary ice (L⁻¹)','Liquid water path (g m⁻²)']):
        a.set(title=title,xlabel='Elapsed time (h)',ylabel=ylab,xlim=(0,6));a.grid(axis='y',alpha=.18)
    ax[0].legend(fontsize=8,frameon=False)
    c=pd.read_csv(p/'contrasts.csv');c=c[c.metric=='primary'].set_index('comparison')
    names=['inactive','quarter','third','half','fresh'];x=[0,25,100/3,50,100]
    e=c.loc[[k+' vs aged' for k in names]]; centers=e.difference.to_numpy()
    ax[2].errorbar(x,centers,yerr=[centers-e.low.to_numpy(),e.high.to_numpy()-centers],
                   c=BLUE,marker='o',capsize=3)
    ax[2].axhline(0,c=INK,lw=.8)
    ax[2].set(title='Activity of incoming carriers',xlabel='Share with fresh-extract activity (%)',
              ylabel='Extra primary ice vs survivor-like inflow (L⁻¹)',xticks=[0,25,50,75,100])
    fig.savefig(p/'cloud_response.png',dpi=175);plt.close(fig)
    fields=pd.read_csv(p/'fields.csv');g=fields[fields.t_s>=4*3600].groupby('cell').liquid_g_m3.mean().to_numpy().reshape(10,10)
    ev=pd.read_csv(p/'freezing_ids.csv');paths=pd.read_csv(p/'tagged_paths.csv.gz')
    fig=plt.figure(figsize=(11.8,5.25),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=[1.2,1]);a=fig.add_subplot(gs[:,0])
    mesh=a.pcolormesh(np.linspace(0,1600,11),np.linspace(0,600,11),g.T,cmap='Blues',shading='flat')
    fig.colorbar(mesh,ax=a,orientation='horizontal',shrink=.8,pad=.10,label='Mean liquid water, hours 4–6 (g m⁻³)')
    x,z=np.meshgrid(np.arange(80,1600,160),np.arange(30,600,60))
    k=2*np.pi/1600;eta=z/600
    u=-np.sin(k*x)*(.55*np.pi*np.cos(np.pi*eta)+.27*eta**2)/(k*600)
    w=np.cos(k*x)*(.55*np.sin(np.pi*eta)+.09*eta**3)
    a.quiver(x,z,u,w,angles='xy',scale_units='xy',scale=.015,color='#38485c',width=.0032,alpha=.63)
    sub=ev[ev.first_freeze_s>=4*3600]
    for origin,marker,c in [('resident','o',RUST),('incoming','^',TEAL)]:
        v=sub[sub.source==origin]
        a.scatter(v.x_m,v.z_m,s=13,marker=marker,c=c,alpha=.58,label=origin.capitalize()+' first freeze')
    a.set(title='Eddy flow and freezing locations',xlabel='Horizontal distance (m)',ylabel='Height (m)',
          xlim=(0,1600),ylim=(0,600))
    a.legend(loc='lower right',fontsize=8,framealpha=.94)
    rgba=['#6a707a',TEAL,BLUE]
    for j,tag in enumerate(['resident','incoming']):
        origin=paths.birth_s<0 if j==0 else paths.birth_s>=0
        ids=paths[origin].id.unique();ident=sorted(ids)[len(ids)//2]
        g=paths[(paths.id==ident)&(paths.fate==1)].copy()
        b=fig.add_subplot(gs[j,1])
        points=g[['time_s','z_m']].to_numpy();points[:,0]/=3600
        b.add_collection(LineCollection(np.stack([points[:-1],points[1:]],axis=1),
                             colors=[rgba[int(v)] for v in g.phase.iloc[:-1]],linewidths=2.6))
        first=g.first_ice_s.max()
        if first>=0:
            erow=ev[ev.id==ident].iloc[0]
            b.scatter(first/3600,erow.z_m,marker='*',c=RUST,s=88,zorder=4,label='First ice')
        b.set(xlim=(0,6),ylim=(0,620),title=f'{tag.capitalize()} particle {ident}',ylabel='Height (m)')
        if j: b.set_xlabel('Time since start (h)')
        b.grid(axis='y',alpha=.16)
        if j==0:
            handles=[Line2D([0],[0],color=c,lw=2.5,label=l) for c,l in zip(rgba,['Dry / haze','Liquid','Ice'])]
            b.legend(handles=handles,loc='lower right',ncol=3,fontsize=8,framealpha=.94)
    fig.savefig(p/'tagged_eddy.png',dpi=175);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('results',type=Path)
    p=ap.parse_args().results;plot(p,write_results(p))
