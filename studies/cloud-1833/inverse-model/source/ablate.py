"""Refit with one mechanism fixed, then test its spectral effect stage by stage."""
from fit import *
from multibox import many
def main():
    start=time.monotonic()
    fits=pd.read_csv(OUT/'frozen.csv')
    records=[];scores=[]
    for entry,grp in fits.groupby('entry'):
        row=grp.loc[grp.train_mse.idxmin()];k=int(row.k)
        for name,fixed in [('unrestricted',{}),('hard_only',{0:0.}),('original_particle_wall',{1:np.log(.5)}),
                           ('reference_production',{2:0.}),('reference_acid_wall',{3:0.})]:
            p=row[NAMES].to_numpy(float).copy();free=np.array([j for j in range(6 if entry else 4) if j not in fixed])
            for j,x in fixed.items():p[j]=x
            cache={}
            def evalx(x):
                key=x.tobytes()
                if key not in cache:
                    p[free]=x
                    s,q=run(96,k,"train",entry,False,p,True)
                    r,j=features(s,q,'train',True)
                    cache.clear();cache[key]=r,j[:,free].copy()
                return cache[key]
            opt=least_squares(lambda x:evalx(x)[0],p[free],jac=lambda x:evalx(x)[1],
                bounds=(LOW[free],HIGH[free]),max_nfev=14,ftol=1e-5,gtol=1e-5,xtol=1e-5)
            p[free]=opt.x
            print('frozen restriction',entry,name,p,flush=True)
            records.append(dict(k=k,entry=entry,restriction=name,train_mse=2*opt.cost,nfev=opt.nfev,**dict(zip(NAMES,p))))
            print('trained',entry,name,flush=True)
    jobs=[(r["k"],r["entry"],0,80,np.array([r[key] for key in NAMES])) for r in records]
    ix=segments['test'];ok=~missing[ix]
    for n in [192,288]:
        for r,(spec,diag) in zip(records,many(n,'test',jobs)):
            for m in metrics(spec[ok],ix[ok]):
                mask=(stage[ix]==m['stage']) if m['stage'] else (stage[ix]<18)
                mask[0]=False;mask &= np.isfinite(frame.ufra_raw.to_numpy()[ix])
                scores.append(dict(k=r['k'],entry=r['entry'],restriction=r['restriction'],n=n,acid_mae=np.mean(abs(diag[mask,0]-frame.measured_cal.to_numpy()[ix[mask]])),**m))
    pd.DataFrame(records).to_csv(OUT/'restriction-fits.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'restriction-transfer.csv',index=False)
    (OUT/'restriction-runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,starts=len(records),fine_replays=20,selection='lowest 64-cell Hg-Xe cost among the five primary treatments separately for each particle inlet')))
if __name__=='__main__':main()
