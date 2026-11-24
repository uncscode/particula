# %% Blind recovery of independently simulated single and paired mechanism changes.
import itertools,json,time
import numpy as np
import pandas as pd
from fit import *
FACTORS=["ideal_hydration","sizing_drying","strong_capture","weaker_wall","more_production","slower_gas_loss","two_regions","predicted_entry"]
DESIGNS=[()]+[(v,) for v in FACTORS]+[pair for pair in itertools.combinations(FACTORS,2) if pair!=tuple(FACTORS[:2])]
RADIUS=np.array([.065,.12,.052,.06,.24,.24])
def config(labels):
    k=0 if FACTORS[0] in labels else 2 if FACTORS[1] in labels else 1
    p=np.array([.87 if FACTORS[2] in labels else .38,np.log(.35 if FACTORS[3] in labels else .65),
        np.log(1.13 if FACTORS[4] in labels else .88),np.log(.72 if FACTORS[5] in labels else 1.03),-.2,.1])
    return k,int(FACTORS[7] in labels),int(FACTORS[6] in labels),p

def transform(s,q,ix): # Number-weighted bands and measured gas, with analytic chain rule.
    edges=np.linspace(e[4],e[-1],9)
    w=overlap(edges[:-1],edges[1:],e)/np.diff(edges)[:,None]
    values=np.concatenate([s@w.T,q[:,:,:,:1]],axis=-1)
    floor=np.r_[np.ones(8),1e5]
    out=np.empty_like(values);out[:,0]=np.log10(floor+values[:,0]);out[:,1:]=values[:,1:]/((floor+values[:,:1])*np.log(10))
    return out
def stack(n,labels,p,ad=False):
    k,entry,two,_=config(labels);ys=[]
    for seg in ["train","test"]:
        ix=segments[seg];s,q=run(n,k,seg,entry,two,p,ad)
        ys.append(transform(s,q,ix))
    return np.concatenate(ys,axis=2)
def label(case):return "+".join(case) if case else "baseline"
def main():
    tic=time.monotonic();rng=np.random.default_rng(183302)
    # Known discrete labels are used only by the generator and the final scoring.
    template=[];jac=[]
    for labels in DESIGNS:
        p=config(labels)[-1];res=stack(64,labels,p,True)[0]
        template.append(res[0]);jac.append(res[1:]*RADIUS[:,None,None])
    template=np.array(template);jac=np.array(jac)
    tr=segments["train"];te=segments["test"];ix=np.r_[tr,te]
    observed=transform(y[None,None],frame.measured_cal.to_numpy()[None,None,:,None],np.arange(len(t)))[0,0]
    valid=np.ones((len(ix),9),dtype=bool);valid[:,8]=np.isfinite(frame.ufra_raw.to_numpy()[ix]);valid[missing[ix],:8]=False
    valid[[0,len(tr)]]=False
    # Anchor the noise model to the best UVH fit in this 64-cell prediction space.
    fits=pd.read_csv(OUT/"frozen.csv");row=fits.loc[fits.train_mse.idxmin()]
    p=row[NAMES].to_numpy(float);s,q=run(64,int(row.k),"train",int(row.entry),False,p)
    noise=observed[tr]-transform(s,q,tr)[0,0]
    for j in range(9):noise[:,j]-=np.mean(noise[valid[:len(tr),j],j])
    scoredef=[]
    for mask in [ix<0,np.ones(len(ix),bool)]:
        if not mask.any():mask[:len(tr)]=True
        mask &= (np.arange(len(ix))%3==0)&(stage[ix]<18)
        weights=np.ones((len(ix),9));weights[:,:8]/=8**.5
        choose=valid&mask[:,None]
        scoredef.append((choose,weights[choose]))
    cases=[];ys=[];ns=12
    for group,labels in enumerate(DESIGNS):
        p=config(labels)[-1];pars=p+(rng.uniform(-.85,.85,(ns,6))*RADIUS)
        truth=stack(80,labels,pars)[:,0]
        for j in range(ns):
            starts=rng.integers(0,len(tr)-9,(len(ix)+8)//9)
            seq=(starts[:,None]+np.arange(9)).ravel()[:len(ix)]
            # Nine consecutive two-minute residuals, all channels drawn jointly.
            ys.append(truth[j]+noise[seq])
            cases.append((group,j))
    ys=np.array(ys); records=[];rankings=[]
    for window,(choose,weights) in zip(["training_only","both_lamps"],scoredef):
        target=ys[:,choose]*weights
        losses=[];solutions=[]
        for mean,J in zip(template,jac):
            X=(J[:,choose]*weights).T
            r=target-mean[choose]*weights
            proj=r@X;gram=X.T@X;coef=np.zeros((len(ys),6))
            # Box-constrained tangent refit within each discrete mechanism.
            for j in range(36):
                for h in range(6):
                    if gram[h,h]>1e-12:
                        coef[:,h]=np.clip((proj[:,h]-coef@gram[:,h]+coef[:,h]*gram[h,h])/gram[h,h],-1,1)
            residual=r-coef@X.T;losses.append(np.mean(residual**2,1));solutions.append(coef)
        losses=np.array(losses).T;order=np.argsort(losses,axis=1)
        for j,(group,rep) in enumerate(cases):
            best,second=order[j,:2]
            records.append((window,rep,label(DESIGNS[group]),label(DESIGNS[best]),label(DESIGNS[second]),int(group==best),losses[j,best],losses[j,group]))
        # Validate top assignments by exact nonlinear 80-cell replays of each
        # truth, winner and runner-up for one preselected replicate per truth.
        if window=="both_lamps":
            for j,(group,rep) in enumerate(cases):
                if rep!=0:continue
                matches={}
                for ci in np.unique([group,*order[j,:2]]):
                    labels=DESIGNS[ci];p=config(labels)[-1]+RADIUS*solutions[ci][j]
                    z=stack(80,labels,p)[0,0]
                    matches[ci]=np.mean(((z[choose]-ys[j,choose])*weights)**2)
                best=min(matches,key=matches.get)
                rankings.append(dict(truth=label(DESIGNS[group]),linear_best=label(DESIGNS[order[j,0]]),exact_subset_best=label(DESIGNS[best]),truth_best=int(best==group)))
    df=pd.DataFrame(records,columns=["window","replicate","truth","selected","runner_up","exact","selected_loss","truth_loss"])
    df.to_csv(OUT/"blind-assignments.csv",index=False);pd.DataFrame(rankings).to_csv(OUT/"nonlinear-recovery-check.csv",index=False)
    rows=[]
    for window,g in df.groupby('window'):
        for factor in FACTORS:
            true=g.truth.map(lambda x:factor in x.split('+'));pred=g.selected.map(lambda x:factor in x.split('+'))
            rows.append(dict(window=window,factor=factor,present=int(true.sum()),detected=int((true&pred).sum()),absent=int((~true).sum()),false_positive=int((~true&pred).sum())))
    pd.DataFrame(rows).to_csv(OUT/"mechanism-recovery.csv",index=False)
    (OUT/"recovery.json").write_text(json.dumps(dict(seconds=time.monotonic()-tic,truth_classes=len(DESIGNS),independent_synthetic_records=len(ys),
        generator_cells=80,fit_cells=64,seed=183302,block_minutes=18,template_tangent_parameters=6,nuisance_half_ranges=RADIUS.tolist(),exact_subset_checks=len(rankings)),indent=2))
    print(df.groupby('window').exact.agg(['sum','count']),time.monotonic()-tic,flush=True)
if __name__=='__main__':main()
