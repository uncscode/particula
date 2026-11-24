"""Simulate the explicit rate grid and synthetic chamber truths."""
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import experiment as chamber

out=Path(sys.argv[2])
scratch=Path(sys.argv[3])
scratch.mkdir(exist_ok=True,parents=True)
out.mkdir(exist_ok=True,parents=True)
stages=[7,13,16,18]
rates=[.015,.025,.035,.05,.065,.085,.11,.15,.20,.27]
flows=[.80,.85,.90,.95,.975,1.,1.025,1.05,1.10,1.15,1.20]
speeds=[.75,.85,1.,1.15,1.25]
cols=['eddy','dilution','low_capture']
grid=pd.DataFrame(itertools.product(rates,flows,speeds),columns=cols)
known=pd.DataFrame(list(itertools.product([.05,.15],[.9,1.,1.1],[1.]))+
                   list(itertools.product([.05,.15],[1.],[.85,1.15]))+
                   [(.06,.97,.93),(.135,1.03,1.07)],columns=cols)
select=chamber.d>=7.6
book=[]
def simulate(config,stage,n,label):
    start=time.monotonic()
    frames,metrics=chamber.calc(config,str(stage),n)
    for m in metrics:assert m['budget_residual']<1e-9
    book.append(dict(label=label,window=stage,cells=n,segments=len(config),
                     max_budget_error=max(m['budget_residual'] for m in metrics),
                     seconds=time.monotonic()-start))
    return frames,metrics
def predict(table,n,label):
    for st in stages:
        pieces=[]
        for pos in range(0,len(table),32):
            frames,_=simulate(table.iloc[pos:pos+32].to_dict('records'),st,n,label)
            pieces.extend(f.attrs['channel_counts'][0][1:,select] for f in frames)
        np.save(scratch/f'{label}-{st}.npy',np.array(pieces))
        if label=='bank':
            f=frames[0];ids=chamber.parts(str(st))[1:]
            mask=~(chamber.missing[ids,None] & (np.flatnonzero(select)[None,:]<7))
            np.savez(scratch/f'obs-{st}.npz',counts=f.attrs['channel_counts'][1][1:,select],mask=mask)
        print(label,st,len(pieces),flush=True)
if __name__=='__main__':
    if sys.argv[4]=='bank':
        ref,_=simulate([dict(eddy=.05)],18,96,'pilot')
        grid.to_csv(scratch/'grid.csv',index=False);known.to_csv(scratch/'truth.csv',index=False)
        np.save(scratch/'diameters.npy',chamber.d[select])
        predict(grid,192,'bank');predict(known,288,'truth')
        pd.DataFrame(book).to_csv(out/'bank-ledger.csv',index=False)
    elif sys.argv[4]=='transfer':
        frozen=pd.concat([pd.read_csv(out/f) for f in ('frozen.csv','shape-frozen.csv')])
        unique=frozen[cols].drop_duplicates()
        # Fine-grid dark refit on the same complete grid is run by the fine mode.
        series=[];metrics=[]
        for n in (192,288):
            fs,ms=simulate(unique.to_dict('records'),'X',n,'illuminated transfer')
            for (_,config),f,m in zip(unique.iterrows(),fs,ms):
                meta=dict(config)|{'cells':n}
                for v in [1000,5000]:
                    for samples in [1,3]:
                        times=[]
                        for col in ['observed','predicted']:
                            ok=(f[col].to_numpy()>=v).astype(int)
                            hits=np.flatnonzero(np.convolve(ok,np.ones(samples),'valid')==samples)
                            times.append(f.archive_h.iloc[hits[0]] if len(hits) else np.nan)
                        m[f'cross_{v}_{samples}_min']=60*(times[1]-times[0])
                metrics.append(meta|m);w=f[['archive_h','stage','predicted','observed','model_small','obs_small','model_middle','obs_middle']].copy()
                for k,v in meta.items():w[k]=v
                w.attrs={};series.append(w)
        pd.DataFrame(metrics).to_csv(out/'transfer.csv',index=False)
        pd.concat(series).to_csv(out/'transfer-series.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        pd.DataFrame(book).to_csv(out/'transfer-ledger.csv',index=False)
    elif sys.argv[4]=='fine':
        predict(grid,288,'fine');pd.DataFrame(book).to_csv(out/'fine-ledger.csv',index=False)
