# %% Exact frozen forward replays; all discrete configurations are retained.
from fit import *
from multibox import many
def main():
    start=time.monotonic();fits=pd.read_csv(OUT/"frozen-final.csv");rows=[];summaries=[];arrays={};gases={};accounts=[]
    if len(CASES)==5:CASES.append(("acidic",1.))
    fits=pd.concat([fits,pd.read_csv(OUT/"acidic-frozen.csv")],ignore_index=True)
    for seg,n in [("train",192),("train",288),("test",192),("test",288)]:
        jobs=[]
        for row in fits.itertuples():
            k,entry=int(row.k),int(row.entry);p=np.array([getattr(row,name) for name in NAMES])
            regimes=[(0,80)] if seg=="train" else [(0,80),(1,60),(1,80),(1,100)]
            jobs.extend((k,entry,two,tau,p) for two,tau in regimes)
        ix=segments[seg]
        predictions=many(n,seg,jobs)
        for (k,entry,two,tau,p),(spec,diag) in zip(jobs,predictions):
            key=f"{seg}_{n}_{k}_{entry}_{two}_{tau}";ok=~missing[ix]
            for m in metrics(spec[ok],ix[ok]):
                summaries.append(dict(key=key,k=k,entry=entry,two=two,tau=tau,n=n,segment=seg,**m))
            if seg=="test":
                okgas=np.isfinite(frame.ufra_raw.to_numpy()[ix])&(stage[ix]<18);okgas[0]=False
                valid=(stage[ix]<18);valid[0]=False;bb=spec[valid,3];bo=y[ix[valid],3];direct=frame.measured_cal.to_numpy()[ix][okgas]
                accounts.append(dict(key=key,acid_bias=np.mean(diag[okgas,0]-direct),acid_mae=np.mean(abs(diag[okgas,0]-direct)),
                    acid_mean=np.mean(diag[okgas,0]),boundary_pred=bb.mean(),boundary_obs=bo.mean(),boundary_mae=np.mean(abs(bb-bo)),boundary_factor=10**np.mean(abs(np.log10(1+bb)-np.log10(1+bo))),
                    entry_mean=diag[valid,1].mean(),resolved_sink_mean=diag[valid,2].mean(),late_dark_tail=np.mean(spec[missing[ix]]@np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7))))))
            if n==288 and tau==80:
                arrays[key]=spec.astype('f4');gases[key]=diag
        print("transfer",seg,n,round(time.monotonic()-start,1),flush=True)
    pd.DataFrame(summaries).to_csv(OUT/"transfer.csv",index=False)
    pd.DataFrame(accounts).to_csv(OUT/"independent-channels.csv",index=False)
    np.savez_compressed(OUT/"final-spectra.npz",**arrays);np.savez_compressed(OUT/"final-gas.npz",**gases)
    (OUT/"transfer-runtime.json").write_text(json.dumps(dict(seconds=time.monotonic()-start,boxes=len(fits)*10,device=wp.get_device(dev).name)))
if __name__=='__main__':main()
