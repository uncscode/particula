# %%
"""Render the cloud-led manuscript figures from the compact, archived run ledger."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from particula.plotting.toc import add_toc_logo
from campaign import SAMPLES, DUST, BASE_SEED, Experiment
from summarize import load, lookup, paired, EXPERIMENT

INK='#202d39'; TEAL='#007e77'; RUST='#a54a24'; BLUE='#385e83'; GRAY='#63727d'
LABELS={'Marine':'Marine','MineralDust':'Mineral dust','ContinentalPristine':'Continental pristine',
        'ContinentalPolluted':'Continental polluted','CombustionDominated':'Combustion',
        'RuralContinental':'Rural continental'}
NAMES={'reference':'Reference eddy','vigorous_dry':'Vigorous, dry eddy','isdac':'ISDAC profile','mpace':'M-PACE profile',
       'cold_deep':'Cold, deep layer','weak_dry':'Weak, dry eddy','moist_mixed':'Moist, mixed layer','warm_thin':'Warm, thin layer'}


def source_figure(out,c):
    figure,axes=plt.subplots(2,2,figsize=(10.1,6.3),layout='constrained',sharex=True,sharey=True)
    for i,(ax,sky) in enumerate(zip(axes.flat,['reference','vigorous_dry','isdac','mpace'])):
        g=c[(c.sky==sky)&(c.grid==10)&(c.metric=='primary')]
        for name,dy,color,marker in [('closed',.13,TEAL,'o'),('old_dust',-.13,RUST,'s')]:
            v=g[g.comparison==name].set_index('source').reindex(SAMPLES)
            if v.delta.isna().all():continue
            ax.errorbar(v.delta,np.arange(len(v))+dy,xerr=[(v.delta-v.lo).clip(lower=0),(v.hi-v.delta).clip(lower=0)],
                        fmt=marker,color=color,markersize=5,elinewidth=1.3,capsize=2.4,
                        mfc=color if name=='closed' else 'white')
        ax.axvline(0,color=INK,lw=1);ax.set_title(NAMES[sky],loc='left')
        ax.set_yticks(np.arange(6),[LABELS[s] for s in SAMPLES]);ax.set_ylim(5.65,-.65)
        ax.grid(axis='y',alpha=.12);ax.set_xlim(-.075,.33)
        if i>1:ax.set_xlabel('Change in primary freezing events (L⁻¹)')
    handles=[Line2D([],[],marker='o',color=TEAL,ls='',label='Compared with closed top'),
             Line2D([],[],marker='s',mfc='white',color=RUST,ls='',label='Compared with survivor-like dust inflow')]
    figure.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False)
    figure.savefig(out/'source-cloud.png',dpi=190);plt.close(figure)


def mean_liquid_history(ledger):
    opener=gzip.open if ledger.suffix=='.gz' else open
    with opener(ledger,'rt') as f:
        histories=[]
        for line in f:
            if not line.endswith('\n'):
                continue
            record=json.loads(line)
            e=Experiment(**record['experiment'])
            if e.sky not in ('mpace','isdac') or e.donor not in (DUST,'closed') or e.seed<BASE_SEED+10000019:
                continue
            if e==Experiment(e.sky,e.donor,e.seed):
                histories.append((record['experiment'],record['evolution']))
    h=pd.DataFrame([dict(sky=e['sky'],donor=e['donor'],**row) for e,ys in histories for row in ys])
    return h.groupby(['sky','donor','t_h'],as_index=False).LWP.mean()


def water_figure(out,all_pairs,d,ledger):
    c=all_pairs[all_pairs.abundance=='equal']
    fig,ax=plt.subplots(1,3,figsize=(13.3,4.0),layout='constrained')
    prof=pd.read_csv(out/'profiles.csv')
    for case,col,ls in [('ISDAC',RUST,'--'),('M-PACE',TEAL,'-')]:
        z=prof[(prof.profile==case)&prof.model_z_m.between(0,600,inclusive='left')]
        ax[0].plot(z.liquid_g_m3,z.model_z_m,c=col,ls=ls,lw=2.4,label=case)
    ax[0].set(title='Initial liquid layer',xlabel='Liquid water (g m⁻³)',ylabel='Height in eddy (m)',ylim=(0,600))
    ax[0].legend(frameon=False)
    h = mean_liquid_history(ledger) if ledger else pd.read_csv(out/'liquid-history.csv')
    for sky,col in [('isdac',RUST),('mpace',TEAL)]:
        for donor,ls in [(DUST,'-'),('closed',':')]:
            if not h.empty:
                y=h[(h.sky==sky)&(h.donor==donor)].groupby('t_h').LWP.mean()
                ax[1].plot(y.index,y,c=col,ls=ls,lw=2.2)
        points=[]
        for mm in [1.2,4.,7.2,12.,20.,90/np.pi]:
            name=sky if mm==90/np.pi else sky+'_we'+str(mm).replace('.','p')
            g=c[(c.sky==name)&(c.comparison=='closed')&(c.grid==10)&(c.source==DUST)&(c.metric=='primary')]
            if len(g):points.append([mm,g.delta.iloc[0],g.lo.iloc[0],g.hi.iloc[0]])
        if points:
            v=np.array(points)
            ax[2].errorbar(v[:,0],v[:,1],yerr=[v[:,1]-v[:,2],v[:,3]-v[:,1]],color=col,fmt='o-',ms=4,capsize=2)
        filled=[]
        lower=4.125 if sky=='isdac' else 8.2
        for mm in [lower,90/np.pi]:
            name=sky if mm==90/np.pi else sky+'_we'+str(mm).replace('.','p')
            g=all_pairs[(all_pairs.sky==name)&(all_pairs.comparison=='closed')&(all_pairs.grid==10)&
                        (all_pairs.source==DUST)&(all_pairs.metric=='primary')&(all_pairs.abundance=='air_volume')]
            if len(g):filled.append([mm,g.delta.iloc[0],g.lo.iloc[0],g.hi.iloc[0]])
        if filled:
            v=np.array(filled)
            ax[2].errorbar(v[:,0],v[:,1],yerr=[v[:,1]-v[:,2],v[:,3]-v[:,1]],fmt='D',ms=5,mfc='white',mec=col,ecolor=col,capsize=3)
    ax[2].legend(handles=[Line2D([],[],c=INK,marker='o',ms=4,label='Equal loading'),
                          Line2D([],[],c=INK,marker='D',mfc='white',ls='',ms=5,label='Filter volumes')],frameon=False,fontsize=8)
    ax[1].set(title='Liquid remaining',xlabel='Elapsed time (h)',ylabel='Liquid water path (g m⁻²)',xlim=(0,6))
    ax[1].legend(handles=[Line2D([],[],color=INK,label='Fresh dust, open top'),
                          Line2D([],[],color=INK,ls=':',label='Closed top')],frameon=False,fontsize=8)
    ax[2].axhline(0,color=INK,lw=.9)
    ax[2].set(title='Effect of exchanging cloud air',xlabel='Mean inward air speed (mm s⁻¹)',ylabel='Fresh dust minus closed top (events L⁻¹)')
    for a in ax:a.grid(axis='y',alpha=.15)
    fig.savefig(out/'water-and-exchange.png',dpi=190);plt.close(fig)


def mineral_means(d):
    rows = []
    for donor in ['grain_feldspar','grain_quartz']:
        for sky in ['reference','cold_deep','very_cold']:
            x = lookup(d,sky=sky,donor=donor)
            spaced=x[x.seed>=BASE_SEED+10000019]
            if len(spaced):
                x=spaced
            if len(x):
                rows.append(dict(sky=sky,donor=donor,**x[['fresh_primary','suspended']].mean()))
    return pd.DataFrame(rows)


def mineral_figure(out,d):
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.6),layout='constrained')
    table = mineral_means(d) if d is not None else pd.read_csv(out/'mineral-means.csv')
    for case,col,marker in [('grain_feldspar',RUST,'o'),('grain_quartz',TEAL,'s')]:
        group = table[table.donor == case]
        for ax,k in zip(axs,['fresh_primary','suspended']):
            ax.plot(range(len(group)),group[k],marker=marker,c=col,lw=1.8,
                    label='Quartz' if case.endswith('quartz') else 'K-feldspar')
    for ax in axs:
        ax.set(xticks=[0,1,2],xticklabels=['Reference','Cold, deep','Colder eddy'],yscale='log')
        ax.grid(axis='y',alpha=.18)
    axs[0].set(title='Ice formed on incoming grains',ylabel='Primary freezing events (L⁻¹)')
    axs[1].set(title='All ice still airborne at six hours',ylabel='Suspended ice (L⁻¹)')
    axs[0].legend(frameon=False);fig.savefig(out/'mineral-grains.png',dpi=190);plt.close(fig)


def supply_figure(out,summary):
    fig,axs=plt.subplots(1,3,figsize=(11.6,4.15),layout='constrained',sharey=True)
    fig.get_layout_engine().set(wspace=.09)
    volume=pd.read_csv(out/'air-volume.csv').set_index('source').loc[SAMPLES]
    y=np.arange(len(SAMPLES))
    axs[0].barh(y,volume.litres_per_well,color=BLUE,height=.52)
    axs[0].set(title='Air per well', xlabel='Sampled volume\n(L per well)')
    axs[1].barh(y,volume.entry_per_litre,color=BLUE,height=.52)
    axs[1].set(title='Liquid wells at hold entry', xlabel='Well equivalents\nper L of sampled air')
    arrivals=summary[(summary.sky=='reference') & (summary.metric=='fresh_primary') &
                    (summary.grid==10) & (summary.comparison=='closed') &
                    (summary.abundance=='air_volume')].set_index('source').loc[SAMPLES]
    np.testing.assert_allclose(arrivals.control, 0.)
    axs[2].errorbar(arrivals.value,y,xerr=[arrivals.value-arrivals.lo,arrivals.hi-arrivals.value],
                    fmt='o',color=TEAL,capsize=2)
    axs[2].set(title='Ice formed on arrivals', xlabel='Six-hour freezing events\nper L of cloud air',xlim=(0,None))
    axs[0].set_yticks(y,[LABELS[s] for s in SAMPLES]);axs[0].set_ylim(5.65,-.65)
    for a in axs:a.grid(axis='x',alpha=.15)
    fig.savefig(out/'sample-volume.png',dpi=190);plt.close(fig)


def toc(out, c):
    clouds = [('reference', 'Reference eddy'), ('isdac', 'ISDAC profile'),
              ('mpace', 'M-PACE profile')]
    with plt.style.context('particula.toc'):
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.subplots_adjust(left=.37, right=.97, top=.66, bottom=.26)
        fig.text(.05, .89, 'Fresh dust inflow', fontsize=12, weight='bold')
        fig.text(.05, .81, 'Compared with a closed cloud top', fontsize=8.6)
        fig.text(.05, .692, 'Simulated case', fontsize=7.3, weight='bold')
        ax.text(-.025, 1.08, 'Fewer freezes', ha='center', fontsize=7.2,
                color=RUST, transform=ax.get_xaxis_transform())
        ax.text(.052, 1.08, 'More freezes', ha='center', fontsize=7.2,
                color=TEAL, transform=ax.get_xaxis_transform())
        values = []
        for sky, _ in clouds:
            rows = c[(c.sky == sky) & (c.grid == 10) & (c.abundance == 'equal') &
                     (c.source == DUST) & (c.metric == 'primary') & (c.comparison == 'closed')]
            values.append(rows.delta.item())
        ax.barh(range(3), values, color=[TEAL if v >= 0 else RUST for v in values], height=.48)
        ax.set_yticks(range(3), [label for _, label in clouds], fontsize=7.5)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.tick_params(axis='x', labelsize=7.5)
        ax.set(ylim=(2.55, -.55), xlim=(-.057, .085), xticks=[-.04, 0, .04, .08])
        ax.axvline(0, lw=1.1, c=INK)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_xlabel('Effect of entrainment on primary ice formation', fontsize=8.1)
        ax.xaxis.set_label_coords(.5, .14, transform=fig.transFigure)
        fig.text(.5, .055, 'Six-hour change (freezing events L⁻¹)', ha='center', fontsize=7.8)
        add_toc_logo(fig, corner='top right')
        fig.savefig(out/'cloud-toc.png', dpi=300)
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('ledger',type=Path);parser.add_argument('out',type=Path)
    a=parser.parse_args()
    plt.rcParams.update({'font.size':10,'axes.titleweight':'bold','axes.spines.right':False,'axes.spines.top':False,
                         'axes.labelcolor':INK,'text.color':INK,'axes.titlecolor':INK,'figure.facecolor':'white'})
    all_pairs=pd.read_csv(a.out/'paired.csv');c=all_pairs.query("abundance=='equal'");d=load(a.ledger).drop_duplicates(EXPERIMENT)
    supply_figure(a.out,all_pairs);source_figure(a.out,c);water_figure(a.out,all_pairs,d,a.ledger);mineral_figure(a.out,d);toc(a.out,c)
