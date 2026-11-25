# %%
"""Challenge fixed per-well rates by letting each well age during a cold exposure."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from challenges import summarize
from study import EDGES, SAMPLES, geometry,read_inputs,train, predictions,score,survival_law

MODELS=[(b,0.) for b in (.25,.35,.45,.50,.55,.60,.65,.70,.85,1.,1.15,1.35)]+[(1.,k) for k in (.15,.25,.4,.6,.8,1.,1.5,2.5,5.)]


def study(pkt,out):
    # See train() for the pooling/geometry contract. The time transform is the
    # only physical change. Reuse the likelihood with explicit optional power.
    raw=geometry(read_inputs(pkt,False)[0]); cv=[]; held=[]
    for name,g in raw.groupby('sample'):
        short=g[g.phase=='short'].copy()
        fold={}
        for _,plate in short.groupby('plate'):
            for i,key in enumerate(sorted(plate.run.unique())):
                fold[key]=i%3
        for b,k in MODELS:
            for j in range(3):
                mask=g.run.map(fold)==j
                fit=g[(g.phase=='ramp')|((g.phase=='short')&~mask)]
                _,rates,prior,_,post=train(fit,power=b,gamma=k)
                omitted=g[mask&(g.h>0)]
                w=post[((omitted.plate-1)*100+omitted.id-1).to_numpy()].copy()
                w[omitted['shift']>80]=prior
                survival=survival_law(EDGES,rates,b,k).T
                distribution=w @ np.column_stack([-np.diff(survival,axis=1),survival[:,-1]])
                idx=np.searchsorted(EDGES[1:],omitted.h,side='left')
                nll=-np.log(np.maximum(distribution[np.arange(len(omitted)),idx],1e-120))
                df=pd.DataFrame(dict(sample=name,plate=omitted.plate,run=omitted.run,
                                     power=b,gamma=k,nll=nll)).groupby(['sample','plate','run','power','gamma']).nll.mean()
                cv.extend(df.reset_index().to_dict('records'))
    c=pd.DataFrame(cv);c.to_csv(out/'duration_short_cv.csv',index=False)
    means=c.groupby(['sample','plate','power','gamma']).nll.mean().groupby(['power','gamma']).mean()
    best=[means[means.index.get_level_values("gamma")==0].idxmin(),means[means.index.get_level_values("gamma")>0].idxmin()]
    for name,g in raw.groupby('sample'):
        for b,k in MODELS:
            fit=train(g,power=b,gamma=k)
            pred=predictions(g,fit,power=b,gamma=k)
            for row in score(pred,str(b)):
                if row['model']=='persistent' and row['subset']=='all':
                    held.append(dict(**row,power=b,gamma=k))
    d=pd.DataFrame(held)
    d.to_csv(out/'duration_overnight.csv',index=False)
    fixed=d[(d.power==1)&(d.gamma==0)].set_index(['sample','plate'])
    summary=[]
    for b,k in MODELS:
        current=d[(d.power==b)&(d.gamma==k)].set_index(['sample','plate'])
        metrics=pd.DataFrame({'gain':fixed.abs_error-current.abs_error,
                              'bias':current.error,'predicted':current.predicted,
                              'early_gain':fixed.brier_early-current.brier_early,'late_gain':fixed.brier_late-current.brier_late}).reset_index()
        summary.extend(summarize(metrics,list(metrics.columns[2:]),f'{b}/{k}'))
    summary=pd.DataFrame(summary);summary['selected']=summary.scope.isin([f'{b}/{k}' for b,k in best])
    summary.to_csv(out/'duration_intervals.csv',index=False)
    print(means);print(summary[summary.selected]);print(d.groupby(['power','gamma'])[['predicted','abs_error','brier_early','brier_late']].mean())

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('input',type=Path);a.add_argument('output',type=Path)
    args=a.parse_args();study(args.input,args.output)
