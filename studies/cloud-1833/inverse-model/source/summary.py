"""Paired stage contrasts, timing, fine-grid diagnostics, and measured transfer plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit import *
def main():
    ix=segments['test'];obs=y[ix];tm=t[ix];st=stage[ix];ill=(st<18);ill[0]=False
    tail=np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))
    dtimes=[tm[np.flatnonzero(obs@tail>=q)[0]] for q in [1000,5000]]
    trans=pd.read_csv(OUT/'transfer.csv');channels=pd.read_csv(OUT/'independent-channels.csv')
    block=trans.query("segment=='test' and stage==0").merge(channels,on='key')
    block['delay1000_min']=(block.first1000-dtimes[0])*60
    block['delay5000_min']=(block.first5000-dtimes[1])*60
    block['stage12_peak_nm']=block.key.map(trans.query("stage==12").set_index('key').mode_nm)
    fits=pd.concat([pd.read_csv(OUT/'frozen-final.csv'),pd.read_csv(OUT/'acidic-frozen.csv')])
    block=block.merge(fits[['k','entry','train_mse']],on=['k','entry'])
    spectra=np.load(OUT/'final-spectra.npz');gas=np.load(OUT/'final-gas.npz')
    for i,r in block.query('n==288 and tau==80').iterrows():
        s=spectra[r.key].astype(float);ds=np.flatnonzero(st==18)
        block.loc[i,'dark_tail_first']=(s@tail)[ds[0]]
        block.loc[i,'dark_tail_last']=(s@tail)[ds[-1]]
    block.to_csv(OUT/'outcomes.csv',index=False)
    rng=np.random.default_rng(183313);draw=rng.integers(0,9,(20000,9))
    stages=np.arange(9,18);weights=np.array([(ill&(st==ss)).sum() for ss in stages])
    def contrast(a,b,label):
        delta=a-b
        samples=(weights[draw]*delta[draw]).sum(1)/weights[draw].sum(1)
        return dict(comparison=label,ratio=10**np.average(delta,weights=weights),p025=10**np.quantile(samples,.025),p975=10**np.quantile(samples,.975),improved=int(sum(delta<0)),stages=len(stages))
    terr=trans.query("segment=='test' and stage>=9 and stage<=17").pivot(index='key',columns='stage',values='log_error')
    act=pd.read_csv(PACKET/'activity/scores.csv')
    mine=[];restr=[]
    for r in block.itertuples():
        base=act.query('section==@r.n and hydration=="original" and retained==1 and capture=="attractive" and segment=="test" and stage>=9 and stage<=17').sort_values('stage').log_error.to_numpy()
        alt=terr.loc[r.key].to_numpy()
        for name,key in [('measured_acid_activity_reference',None),('same_physics_measured_entry',f'test_{r.n}_{r.k}_0_{r.two}_{r.tau}'),('same_physics_uniform_chamber',f'test_{r.n}_{r.k}_{r.entry}_0_80'),('original_wet_same_entry_chamber',f'test_{r.n}_0_{r.entry}_{r.two}_{r.tau}')]:
            mine.append(dict(key=r.key,**contrast(alt,base if key is None else terr.loc[key].to_numpy(),name)))
    pd.DataFrame(mine).to_csv(OUT/'paired-stage-contrasts.csv',index=False)
    selected=fits.loc[fits.groupby('entry').train_mse.idxmin()] if not fits.index.duplicated().any() else fits.sort_values('train_mse').drop_duplicates('entry')
    keys=[f'test_288_{int(r.k)}_{int(r.entry)}_1_80' for r in selected.sort_values('entry').itertuples()]
    a,b=(spectra[key].astype(float) for key in keys);w=np.diff(e)
    wavg=lambda x:np.nanmean(x,axis=0)
    differences=[]
    for ss in stages:
        mask=ill&(st==ss)
        for j in range(3,len(d)):
            delta=np.mean((a[mask,j]-b[mask,j])*w[j]);observed=np.mean(obs[mask,j]*w[j])
            differences.append(dict(stage=ss,diameter_nm=d[j],start_time=tm[mask][0],end_time=tm[mask][-1],mean_count_difference_per_bin=delta,observed_bin_count=observed))
    sep=pd.DataFrame(differences);sep.iloc[np.argsort(-abs(sep.mean_count_difference_per_bin))].to_csv(OUT/'entry-discrimination.csv',index=False)
    colors=['#007570','#b35b19'];labels=['Measured-boundary estimate','Predicted particle entry']
    f,ax=plt.subplots(2,2,figsize=(12,7.9),layout='constrained')
    a,b,c,h=ax.ravel()
    for axis in [a,b,c]:
        axis.axvspan(tm[st==18][0],tm[-1],color='#c4c8d0',alpha=.4)
        axis.set_xlabel('Archive time (h)')
    for threshold in [1000,5000]:a.axhline(threshold,color='#c7cbd0',linewidth=.6,zorder=0)
    observed=obs@tail
    a.plot(tm,observed,color='black',label='Measured',lw=1.6)
    b.plot(tm,np.where(np.isfinite(frame.ufra_raw.to_numpy()[ix]),frame.measured_cal.to_numpy()[ix]/1e7,np.nan),color='black',lw=1.1)
    b.plot(tm,frame.author_chem.to_numpy()[ix]/1e7,color='#777777',ls=':',label='Published chemistry')
    c.plot(tm,np.where(missing[ix],np.nan,obs[:,3]),color='black',lw=1.2)
    for lab,key,col in zip(labels,keys,colors):
        a.plot(tm,spectra[key]@tail,label=lab,color=col)
        b.plot(tm,gas[key][:,0]/1e7,color=col)
        c.plot(tm,spectra[key][:,3],color=col)
    a.set(ylabel='Particles ≥9.7 nm (cm⁻³)',title='Growth through 9.7 nm');a.legend(fontsize=8,frameon=False)
    b.set(ylabel='Sulfuric acid (10⁷ molecules cm⁻³)',title='Gas concentration');b.legend(frameon=False,fontsize=8)
    c.set(ylabel='Particles cm⁻³ per decade',title='Density in the 2.9 nm measurement bin')
    recovery=pd.read_csv(OUT/'mechanism-recovery.csv');m=recovery.query("window=='both_lamps'");idx=np.arange(len(m))
    h.barh(idx-.18,m.detected/m.present,height=.34,color='#007570',label='Detected when present')
    h.barh(idx+.18,m.false_positive/m.absent,height=.34,color='#ae6970',label='Selected when absent')
    h.set(yticks=idx,yticklabels=['Original hydration','Partial drying','Stronger capture','Weaker wall loss','Higher production','Slower gas loss','Two regions','Predicted entry'],xlim=(0,1.03),xlabel='Fraction of synthetic records',title='Blind mechanism recovery, both lamps')
    h.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,frameon=False,fontsize=8)
    f.savefig(OUT/'inverse-results.png',dpi=155);plt.close(f)
    mask=ill&np.isfinite(frame.ufra_raw.to_numpy()[ix]);archive=frame.author_chem.to_numpy()[ix]
    (OUT/'diagnostic-summary.json').write_text(json.dumps(dict(training_selected_two_region_80s_keys=keys,measured_first_hours=dtimes,scored_spectra=int(ill.sum()),
        raw_acid_samples=int(mask.sum()),published_gas_mae=float(np.mean(abs(archive[mask]-frame.measured_cal.to_numpy()[ix[mask]]))),measured_dark_tail=[observed[st==18][0],observed[st==18][-1]],stage_bootstrap_draws=len(draw),seed=183313),indent=2))
    if "--transfer-only" in __import__("sys").argv:return
    rf=pd.read_csv(OUT/'restriction-transfer.csv').query('stage>=9 and stage<=17')
    for (entry,n),g in rf.groupby(['entry','n']):
        tab=g.pivot(index='restriction',columns='stage',values='log_error')
        for name,vals in tab.iterrows():restr.append(dict(entry=entry,n=n,**contrast(tab.loc["unrestricted"].to_numpy(),vals.to_numpy(),name)))
    pd.DataFrame(restr).to_csv(OUT/'paired-restrictions.csv',index=False)
if __name__=='__main__':main()
