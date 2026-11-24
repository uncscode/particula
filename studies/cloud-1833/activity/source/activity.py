# %% Forward comparison, fixed physical flux at the 2.9 nm chamber diameter.
import ctypes, json, os, time
from pathlib import Path
from physics import *
import warp as wp
OUT=Path(os.environ.get("ACT_OUT",PACKET/"activity"))
OUT.mkdir(exist_ok=True,parents=True)
def replay(n,law,retain,idx,legacy=False):
    a=geometry(n,law,retain,legacy);h=t[idx];nt=len(h)
    vel=a["vel"]; configs=np.array([[1.1,.5,2,1],[1.1,.5,1,1]],dtype=float)
    # Existing kernel's three law slots are vapor (unused), attractive, finite-hard.
    v=np.array([vel[0],vel[1],vel[0]])
    measured=frame.measured_cal.to_numpy()[idx];initial=a["mi"]@y[idx[0]]
    ref=geometry(64,legacy=True)["vel"][1,0]*1.1
    args=[h,np.tile(measured,(2,1)),y[idx,3],initial,np.r_[v.ravel(),v.max(1)],a["wall"],configs,a["K"],
          *a["pairs"],a["over"],np.zeros(nt),ref*measured]
    arrays=[wp.array(np.ascontiguousarray(z),device="cuda:0") for z in args]
    out=wp.empty((2,nt,n),dtype=wp.float64,device="cuda:0");ledger=wp.empty((2,n,6),dtype=wp.float64,device="cuda:0")
    lib=ctypes.CDLL(os.environ["CLOUD_SOLVER"]);lib.replay.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*2+[ctypes.POINTER(ctypes.c_void_p)]
    wp.synchronize();ptrs=(ctypes.c_void_p*17)(*[z.ptr for z in arrays+[out,ledger]])
    assert lib.replay(2,n,nt,a["dx"],6 if n==288 else 12,ptrs)==0
    x=out.numpy();l=ledger.numpy().sum(1)
    rel=abs(x[:,-1].sum(1)-(initial.sum()+l[:,0]-l[:,1:].sum(1)))/(initial.sum()+l[:,0])
    assert rel.max()<1e-9 and x.min()>=0 and np.isfinite(x).all()
    return x@a["mo"].T/np.diff(e),1.1*np.einsum("cti,ci->ct",x,a["cap"]),float(rel.max())

def main():
    start=time.monotonic();wp.init(); assert wp.get_cuda_device_count()
    idx=segments["test"];early=idx[~missing[idx]]
    pilot,_,_=replay(64,"original",1.,early[:16],True)
    check,_,_=replay(288,"original",1.,early,True)
    old=np.load(PACKET/"mechanism.npz")["fixed_flux"][[1,2]]
    delta=np.max(abs(check-old)); assert delta<1e-6,delta
    cases=[("original",1.),("aiomfac",1.),("aiomfac",.5),("aiomfac",0.),("dry",1.)]
    rows=[];series=[];spectra={};budget=[];boot=[]
    for n in [192,288]:
        for law,retain in cases:
            for seg,ix in segments.items():
                p,cs,err=replay(n,law,retain,ix);budget.append(err)
                for c,name in enumerate(["hard","attractive"]):
                    key=f"{seg}_{n}_{law}_{retain}_{name}"
                    use=~missing[ix]
                    for m in metrics(p[c,use],ix[use]):
                        rows.append(dict(key=key,section=n,hydration=law,retained=retain,capture=name,segment=seg,**m))
                    if n==288:
                        spectra[key]=p[c].astype("f4")
                        for ti,nu,sink in zip(ix,p[c],cs[c]):
                            series.append([key,t[ti],int(stage[ti]),np.dot(nu,np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))),sink])
    pd.DataFrame(rows).to_csv(OUT/"scores.csv",index=False)
    pd.DataFrame(series,columns=["case","archive_h","stage","tail_cm3","resolved_sink_s"]).to_csv(OUT/"series.csv.gz",index=False)
    # Retain signed residual heatmaps through saved spectra, not raw solver histories.
    np.savez_compressed(OUT/"spectra.npz",**spectra)
    sizes=np.geomspace(2.9,96,250)
    pd.DataFrame({"diameter_nm":sizes,"aw_target":.58*np.exp(-kelvin/sizes),"water_mass_fraction":fraction(sizes,"aiomfac"),
                  "original_water_mass_fraction":fraction(sizes,"original"),"dry_reported_nm":reported(sizes,"aiomfac",0.)}).to_csv(OUT/"activity.csv",index=False)
    (OUT/"freeze.json").write_text(json.dumps(dict(revision="af0dc67bf7",scenarios=cases,decision="Keep all calibrated and uncalibrated measurement endpoints, dry control and intermediate retention; no excimer-based elimination.",cpu_port="direct homogeneous liquid: fixed fully neutralized sulfate",device=wp.get_device("cuda:0").name,cuda_boxes=44,max_balance_error=max(budget),baseline_max_abs_density_difference=delta,elapsed_s=time.monotonic()-start),indent=2))
    print("activity",delta,max(budget),time.monotonic()-start)
if __name__=="__main__":main()
