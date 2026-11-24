# %% Fit Hg-Xe spectra and gas jointly, including differentiation of the live simulation.
import argparse, json, os, time
from pathlib import Path
from scipy.optimize import least_squares
from physics import *
from forward import *
OUT=Path(os.environ.get("INV_OUT",PACKET/"inverse-model"));OUT.mkdir(exist_ok=True,parents=True)
def design(seg):
    ix=segments[seg]
    # Actual measurement widths act as quadrature weights. Gas weight sums to one
    # like the entire spectrum; missing measured acid contributes no residual.
    mask=(np.arange(len(ix))>0)&~missing[ix]
    if seg=="test":mask &= stage[ix]<18
    gasmask=mask&np.isfinite(frame.ufra_raw.to_numpy()[ix])
    return ix,mask,gasmask

def features(spec,diag,seg,deriv):
    ix,m,gm=design(seg);s=spec[0][:,m,4:];g=diag[0][:,gm,0]
    w=np.sqrt(np.diff(e)[4:]/np.diff(e)[4:].sum()/m.sum())
    # log derivatives of positive shifted outputs
    ref=y[ix[m],4:]; gr=frame.measured_cal.to_numpy()[ix[gm]]
    sr=(np.log10(1+s[0])-np.log10(1+ref))*w
    ar=(np.log10(1e5+g[0])-np.log10(1e5+gr))/np.sqrt(gm.sum())
    r=np.r_[sr.ravel(),ar]
    if not deriv:return r
    sj=s[1:]*w/((1+s[0])*np.log(10));aj=g[1:]/((1e5+g[0])*np.log(10)*np.sqrt(gm.sum()))
    return r,np.column_stack([sj.reshape(6,-1),aj]).T

def exactfit(k,entry,start,n,its):
    free=np.arange(6 if entry else 4);last=[None,None,None];calls=0
    def ev(x):
        nonlocal calls
        if last[0] is None or not np.array_equal(x,last[0]):
            p=start.copy();p[free]=x
            s,q=run(n,k,"train",entry,False,p,True)
            r,j=features(s,q,"train",True);last[:]=[x.copy(),r,j[:,free]];calls+=1
        return last[1:]
    opt=least_squares(lambda x:ev(x)[0],start[free],jac=lambda x:ev(x)[1],
        bounds=(LOW[free],HIGH[free]),max_nfev=its,ftol=1e-5,gtol=1e-5,xtol=1e-5)
    p=start.copy();p[free]=opt.x;sv=np.linalg.svd(opt.jac,compute_uv=False)
    return p,float(2*opt.cost),calls,sv,float(opt.optimality)
def runfit():
    start=time.monotonic()
    if not (PACKET/"activity/freeze.json").exists():raise RuntimeError("activity handoff missing")
    initial=[CENTER.copy(),np.array([.18,np.log(.9),np.log(1.14),np.log(1.2),np.log(3.),np.log(.4)])]
    rows=[]
    for k in range(len(CASES)):
        for entry in (0,1):
            for which,x in enumerate(initial):
                p,l,c,s,o=exactfit(k,entry,x,64,18)
                rows.append(dict(k=k,entry=entry,start=which,train_mse=l,grad_calls=c,stationarity=o,min_singular=s[-1],max_singular=s[0],**dict(zip(NAMES,p))))
            pd.DataFrame(rows).to_csv(OUT/"fits.csv",index=False)
            print("fit",k,entry,rows[-2]['train_mse'],rows[-1]['train_mse'],round(time.monotonic()-start,1),flush=True)
    # Freeze candidates using training information exclusively. Replay every discrete
    # alternative at both fine resolutions to retain physically distinct near ties.
    df=pd.DataFrame(rows);fit=df.loc[df.groupby(["k","entry"]).train_mse.idxmin()]
    fit.to_csv(OUT/"frozen.csv",index=False)
    (OUT/"fit-runtime.json").write_text(json.dumps(dict(seconds=time.monotonic()-start,direct_ad_evaluations=sum(r["grad_calls"] for r in rows),starts=len(rows),training_bins=64,shared_training="identical illumination during Hg-Xe makes both regional geometries equivalent")))
def pilot():
    start=time.monotonic();k=1;p=CENTER.copy()
    s,q=run(64,k,"train",1,False,p,True)
    r,j=features(s,q,"train",True);direction=np.array([.08,-.2,.15,.1,-.25,.11])
    errors=[]
    for eps in (1e-4,2e-5):
        sp,qp=run(64,k,"train",1,False,np.stack([p+eps*direction,p-eps*direction]))
        fd=(features(sp[:1],qp[:1],"train",False)-features(sp[1:],qp[1:],"train",False))/(2*eps)
        errors.append(float(np.linalg.norm(fd-j@direction)/np.linalg.norm(fd)))
    # Compare identical illumination two-region and one-region, including growth & gas exchange.
    s2,q2=run(64,k,"train",1,True,p)
    diff=float(np.max(abs(s2[0,0]-s[0,0])))
    assert min(errors)<.005 and diff<1e-6,(errors,diff)
    (OUT/"pilot.json").write_text(json.dumps(dict(device=wp.get_device(dev).name,seconds=time.monotonic()-start,gradient_relative_error=errors,uniform_region_spectrum_max_difference=diff),indent=2))
    print("pilot",errors,diff,time.monotonic()-start,flush=True)
if __name__=="__main__":
    import sys
    {"pilot":pilot,"fit":runfit}[sys.argv[1]]()
