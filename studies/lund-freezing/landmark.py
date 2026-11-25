# %%
"""Forecast a common 6–10 h target from progressive observations on the same plate."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from study import EDGES, geometry,read_inputs,train
from challenges import summarize


def run(pkt,out):
    raw,_,_=read_inputs(pkt,False);raw=geometry(raw)
    scores=[]; err=[]
    for name,sample in raw.groupby('sample'):
        _,rates,prior,_,post=train(sample)
        for plate,night in sample[(sample.phase=='night')&(sample.h>0)].groupby('plate'):
            w=post[((plate-1)*100+night.id-1).to_numpy()].copy()
            w[night['shift']>80]=prior
            t=night.h.to_numpy()
            remaining=t>6
            actual=100*np.mean((t>6)&(t<=10))
            def survival(h,scale):
                return w @ np.exp(-rates*h*scale)
            for cut in [0,2,4,6]:
                # Shared acceleration or retardation, weak N(0,1) prior on log speed.
                # Refit with only the interval between consecutive recorded cutoffs.
                edges=np.r_[EDGES[EDGES<cut],np.arange(3,cut),cut] if cut else np.array([0])
                bucket=np.searchsorted(edges[1:],t,side='left')
                def objective(log_scale):
                    scale=np.exp(log_scale)
                    surv=w @ np.exp(-np.outer(rates,edges*scale))
                    prob=np.column_stack([-np.diff(surv,axis=1),surv[:,-1]])
                    return .5*log_scale**2-np.log(np.maximum(prob[np.arange(len(t)),bucket],1e-120)).sum()
                scale=np.exp(minimize_scalar(objective,bounds=(-4,4),method='bounded').x) if cut else 1
                for model,mult in [('fixed',1),('updated',scale)]:
                    # 6 h count is always known here; same evaluation risk set.
                    prob=1-survival(10,mult)[remaining]/survival(6,mult)[remaining]
                    forecast=100*prob.sum()/len(t)
                    events=(t[remaining]<=10).astype(float)
                    scores.append(dict(sample=name,plate=plate,cut_h=cut,model=model,scale=mult,
                                       observed=actual,predicted=forecast,abs_error=abs(forecast-actual),
                                       brier=np.mean((prob-events)**2),n6=int(remaining.sum())))
    s=pd.DataFrame(scores);s.to_csv(out/'landmarks.csv',index=False)
    for cut in [2,4,6]:
        x=s[s.cut_h==cut].pivot(index=['sample','plate'],columns='model',values=['abs_error','brier']).reset_index()
        f=pd.DataFrame(dict(sample=x['sample'],plate=x['plate'],gain=x.abs_error.fixed-x.abs_error.updated,
                            brier_gain=x.brier.fixed-x.brier.updated))
        err.extend(summarize(f,['gain','brier_gain'],f'{cut} h'))
    pd.DataFrame(err).to_csv(out/'landmark_intervals.csv',index=False)
    print(pd.DataFrame(err))
    # Emulate event-weighted ordinates in a log interval around each endpoint.
    # Bounds use the final complete window ending at the target; no post-hold ramps.
    refs=pd.read_csv(out/'paper_reference.csv')
    convention=[]
    for (name,r),g in raw[raw.phase!='ramp'].groupby(['sample','run']):
        frame=g.sort_values(['seconds','id']).copy()
        frame['ordinate']=frame.seconds.rank(method='max')/100
        for t in [2,10]:
            if t==10 and (frame.phase=='short').all():continue
            for dex in [.05,.1,.2]:
                selected=frame[(frame.h>t/(10**dex))&(frame.h<=t)]
                convention.append(dict(sample=name,run=r,h=t,dex=dex,events=len(selected),
                                       sum_ordinates=selected.ordinate.sum(),fixed=(frame.h<=t).mean()))
    freq=pd.DataFrame(convention).groupby(['sample','h','dex']).agg(events=('events','sum'),total=('sum_ordinates','sum'),fixed=('fixed','mean')).reset_index()
    freq['event_average']=freq.total/freq.events.replace(0,np.nan)
    freq=freq.merge(refs.dropna(subset=['table'])[['sample','h','table']],on=['sample','h'])
    freq.to_csv(out/'reference_weighting.csv',index=False)
    print(freq.query('dex==0.1').round(3).to_string(index=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('input',type=Path);a.add_argument('output',type=Path)
    args=a.parse_args();run(args.input,args.output)
