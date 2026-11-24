"""Correlated synthetic records, label-blind grid retrieval and stagewise transfer."""
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

out=Path(sys.argv[1]);tmp=Path(sys.argv[2])
stages=[7,13,16,18];cols=['eddy','dilution','low_capture']
bounds=[1.,.10,.05,.025,0.]
grid=pd.read_csv(tmp/'grid.csv');truth=pd.read_csv(tmp/'truth.csv')
diams=np.load(tmp/'diameters.npy')
small=diams<9.7
split=np.sqrt(.05*.15)
reps=80
ref_index=grid.query('eddy==0.05 and dilution==1 and low_capture==1').index[0]
def stage_arrays(stage,library):
    o=np.load(tmp/f'obs-{stage}.npz')
    return (np.log(np.load(tmp/f'{library}-{stage}.npy')+2),
            np.log(o['counts']+2),o['mask'])
def discrepancy(pred,measure,mask,center_time=False):
    # pred [candidate,time,diameter], measures [record,time,diameter]
    if center_time:
        pred=pred-(pred*mask).sum(1,keepdims=True)/mask.sum(0)
        measure=measure-(measure*mask).sum(1,keepdims=True)/mask.sum(0)
    result=[]
    for region in [small,~small]:
        valid=mask & region
        a=pred[:,valid];b=measure[:,valid]
        cost=(np.mean(a*a,axis=1)[:,None]+np.mean(b*b,axis=1)[None,:]
              -2*(a@b.T)/a.shape[1])
        result.append(np.maximum(cost,0))
    return np.array(result)

def retrieve(scores,flow_observations,bound,flexible_growth):
    allowed=np.ones_like(scores,dtype=bool)
    if bound!=1.:
        dilution=grid.dilution.to_numpy()[:,None]
        allowed &= np.abs(flow_observations[None,:]/dilution-1)<=bound+1e-11
    if not flexible_growth:allowed &= grid.low_capture.to_numpy()[:,None]==1
    # For off-grid truths exact flow is implemented as nearest simulated flow.
    empty=~allowed.any(axis=0)
    if empty.any():
        delta=np.abs(grid.dilution.to_numpy()[:,None]-flow_observations[None,empty])
        keep=np.isclose(delta,delta.min(axis=0))
        if not flexible_growth:keep &= grid.low_capture.to_numpy()[:,None]==1
        allowed[:,empty]=keep
    return np.where(allowed,scores,np.inf).argmin(axis=0)

def synthetic():
    rng=np.random.default_rng(183304)
    keys=np.repeat(np.arange(len(truth)),reps)
    rng.shuffle(keys)
    # A common random reading error couples the precision scenarios fairly.
    unit_flow_error=rng.uniform(-1,1,len(keys))
    flow=truth.dilution.to_numpy()[keys]
    total={k:0 for k in ['primary','centered','clean','smooth','primary_temporal','smooth_temporal']}
    for st in stages:
        pred,obs,mask=stage_arrays(st,'bank')
        true_log=np.log(np.load(tmp/f'truth-{st}.npy')+2)
        residual=obs-pred[ref_index];nt,nch=residual.shape
        # Fill only a missing donor value. Missing targets are always excluded.
        for c in range(nch):
            ok=mask[:,c]
            if not ok.all():residual[:,c]=np.interp(np.arange(nt),np.flatnonzero(ok),residual[ok,c])
        mean=residual.mean(axis=0)
        centered=residual-mean
        # Moving blocks of four scans preserve joint size-channel fluctuations.
        blocks=rng.integers(0,nt,(len(keys),int(np.ceil(nt/4)),1))
        row=((blocks+np.arange(4))%nt).reshape(len(keys),-1)[:,:nt]
        error=centered[row]
        bias=rng.choice([-1,1],len(keys))[:,None,None]*mean
        # Sensitivity retaining the observed within-stage drift shape.
        trend=pd.DataFrame(residual).rolling(5,center=True,min_periods=1).mean().to_numpy()
        jitter=residual-trend
        smooth_sign=np.random.default_rng(867+st).choice([-1,1],len(keys))[:,None,None]
        smooth=smooth_sign*trend[None,:,:]+jitter[row]
        for name,noise in [('primary',error+bias),('centered',error),('clean',0),('smooth',smooth)]:
            sim=np.maximum(np.log(2),true_log[keys]+noise)
            total[name]+=discrepancy(pred,sim,mask)/4
            if name in ('primary','smooth'):
                total[name+'_temporal']+=discrepancy(pred,sim,mask,center_time=True)/4
    rows=[]
    for noise in total:
        for mode,loss in [('upper',total[noise][1]),('joint',total[noise].mean(axis=0))]:
            for bound in bounds:
                readings=flow*(1+bound*unit_flow_error) if bound<1 else np.zeros_like(flow)
                for flexible in [False,True]:
                    choices=retrieve(loss,readings,bound,flexible)
                    # Join truth only after all choices for this experiment freeze.
                    for key in range(len(truth)):
                        real=truth.iloc[key];chosen=grid.iloc[choices[keys==key]]
                        correct=((chosen.eddy>=split)==(real.eddy>=split)).mean()
                        err=(chosen.eddy/real.eddy-1).abs()
                        rows.append(dict(truth_id=key,**real.to_dict(),noise=noise,channels=mode,
                            precision=bound,flex_growth=flexible,records=len(chosen),correct_wall=correct,
                            absrel50=err.median(),absrel90=err.quantile(.9),fit_eddy50=chosen.eddy.median(),
                            fit_flow50=chosen.dilution.median(),flow_absrel50=(chosen.dilution/real.dilution-1).abs().median(),
                            fit_growth50=chosen.low_capture.median(),correct_growth_direction=(
                            np.sign(chosen.low_capture-1)==np.sign(real.low_capture-1)).mean(),
                            joint_exact=np.all(np.isclose(chosen[cols].to_numpy(),real.to_numpy()),axis=1).mean()))
                    if noise=='primary' and mode=='joint' and not flexible:
                        # Decision pools records within each true rate, not medians of scenario medians.
                        main=(keys<6)
                        for v in [.05,.15]:
                            ind=main & (truth.eddy.to_numpy()[keys]==v)
                            fitted=grid.eddy.to_numpy()[choices[ind]]
                            decisions.append(dict(precision=bound,eddy=v,records=len(fitted),
                                correct_class=np.mean((fitted>=split)==(v>=split)),
                                median_absrel=np.median(abs(fitted/v-1))))
    frame=pd.DataFrame(rows)
    nominal=frame[(frame.low_capture==1)&(frame.truth_id<6)]
    nominal[nominal.precision==1].to_csv(out/'01-blind-recovery.csv',index=False)
    nominal.to_csv(out/'02-flow-precision.csv',index=False)
    frame.to_csv(out/'03-growth-confusion.csv',index=False)
    dec=pd.DataFrame(decisions)
    dec['passes']=(dec.correct_class>=.90)&(dec.median_absrel<=.25)
    dec.to_csv(out/'decision.csv',index=False)
    qualify=[v for v in bounds if dec[dec.precision==v].passes.all()]
    return qualify[0] if qualify else None

def real_folds(qualifier,center_time=False):
    scores=[];choices=[]
    # Retain all flow scenarios for transparency, even when one passes the gate.
    for library in ['bank','fine']:
        stage_sq=[];stage_mae=[]
        for st in stages:
            pred,obs,mask=stage_arrays(st,library)
            stage_sq.append(discrepancy(pred,obs[None,:,:],mask,center_time=center_time)[:,:,0])
            stage_mae.append([np.abs(pred[:,mask&region]-obs[mask&region]).mean(axis=1) for region in (small,~small)])
        sq=np.array(stage_sq);mae=np.array(stage_mae)
        for hold in stages+[0]:
            training=np.array([v!=hold for v in stages])
            for mode,region in [('upper',1),('joint',None)]:
                train=sq[training].mean(axis=0)
                selected=train.mean(axis=0) if region is None else train[region]
                for bound,flex in itertools.product(bounds,[False,True]):
                    k=retrieve(selected[:,None],np.ones(1),bound,flex)[0]
                    row=dict(calibration_cells=192 if library=='bank' else 288,held=hold,
                             channels=mode,precision=bound,flex_growth=flex,selection_sq=selected[k],
                             **grid.iloc[k].to_dict())
                    if hold:
                        sid=stages.index(hold);row.update(small_factor=np.exp(mae[sid,0,k]),large_factor=np.exp(mae[sid,1,k]))
                    choices.append(row)
    pd.DataFrame(choices).to_csv(out/('shape-frozen.csv' if center_time else 'frozen.csv'),index=False)
    return {'decision_precision':qualifier,'noise_replicates_per_truth':reps,'truths':len(truth),
            'blind_records_per_noise_treatment':len(truth)*reps,'precision_encoding':'1 = unconstrained; 0 = exact',
            'seed':183304,'eddy_class_boundary_s_inv':split}
if __name__=='__main__':
    decisions=[]
    selected=synthetic()
    real_folds(selected,center_time=True)
    (out/'decision.json').write_text(json.dumps(real_folds(selected),indent=2)+'\n')
