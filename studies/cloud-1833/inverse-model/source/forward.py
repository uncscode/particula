"""GPU forward model and exact directional AD of each continuous fit parameter."""
import ctypes
from functools import lru_cache
import warp as wp
from physics import *
NAMES=["collision_fraction","log_particle_wall","log_lamp_production","log_acid_wall","log_neutral_birth","log_ion_birth"]
LOW=np.array([0.,np.log(.15),np.log(.65),np.log(.6),np.log(.08),np.log(.08)])
HIGH=np.array([1.,np.log(1.6),np.log(1.35),np.log(1.5),np.log(12.),np.log(12.)])
CENTER=np.array([.8,np.log(.55),np.log(.9),0.,0.,0.])
CASES=[("original",1.),("aiomfac",1.),("aiomfac",.5),("aiomfac",0.),("dry",1.)]
NC=int(os.environ.get("CLUSTER_CELLS","6"))
CED=np.geomspace(.8,2.9,NC+1)
MID=(CED[:-1]*CED[1:])**.5
dev="cuda:0"
def gpu(x):
    return wp.array(np.ascontiguousarray(x),device=dev)
@lru_cache(maxsize=96)
def prepare(n,k,seg,entry):
    law,retain=CASES[k];a=geometry(n,law,retain);idx=segments[seg]
    obs=y[idx[0]].copy()
    if entry:obs[3]=np.interp(np.log(d[3]),np.log(d[[2,4]]),obs[[2,4]])
    amp=frame.production.to_numpy()[idx]
    flags=np.isin(stage[idx],[1,4,6,10,12,15,17]).astype(float)+2*(stage[idx]>=9)
    # A conservative face CFL ceiling using the maximum lamp source / minimum wall loss;
    # guard against large out-of-range concentrations on validation fits.
    ceil=np.full(len(idx),max(2e8,2*1.35*np.max(amp)/(.002*.6+.0001754)))
    f=np.column_stack([t[idx],amp,common_input()[idx] if not entry else np.zeros(len(idx)),frame.measured_cal.to_numpy()[idx],flags,ceil])
    vel=1.1*a["vel"];cap=1.1*a["cap"]
    # One water per neutralized acid unit for newly formed clusters; the 0.8–1 nm
    # endpoint uses the 1 nm capture value of the native table, a stated closure.
    ck=capture(MID)*1.1
    units=1350*np.pi/6*(MID*1e-9)**3/((salt+18.01528)*amu)
    cr=ck/(3*units[:,None].T/MID)/np.diff(CED)
    dw=.002*np.sqrt(par.particles.get_diffusion_coefficient_via_system_state(MID*.5e-9,T,P)/(8e-6*(T/293.15)**1.75))
    # Store hard and attractive cluster sink coefficients separately.
    cl=np.r_[cr.ravel(),dw,ck.ravel()]
    args=[f,a["mi"]@obs,np.r_[vel.ravel(),vel.max()],cap,np.r_[a["wall"],a["over"].astype(float)],a["K"],*a["pairs"],cl]
    if law=="dry": c_map=CED
    else:
        dry=CED*(1350*salt/(1770*(salt+18.01528)))**(1/3)
        c_map=(dry**3+retain*(CED**3-dry**3))**(1/3)
        # Join the instrument diameter map continuously at the 2.9 nm interface.
        c_map*=10**a["mapped"][0]/c_map[-1]
    clmap=overlap(e[:-1],e[1:],np.log10(c_map))/np.diff(np.log10(c_map))
    mapper=np.concatenate((a["mo"],clmap),axis=1)/np.diff(e)[:,None]
    return idx,a,mapper,[gpu(v) for v in args]
def run(n,k,seg,entry,two,params,grad=False,tau=80.,dt=None):
    params=np.atleast_2d(params).astype(float);idx,a,mapper,arrays=prepare(n,k,seg,int(entry));nt=len(idx)
    nd=7 if grad else 1
    p=gpu(params);out=wp.empty((len(params),nd,nt,n+NC+4),dtype=wp.float64,device=dev)
    lib=ctypes.CDLL(os.environ["INVERSE_SOLVER"])
    lib.inverse_run.argtypes=[ctypes.c_int]*6+[ctypes.c_double]*3+[ctypes.POINTER(ctypes.c_void_p)]
    wp.synchronize_device(dev)
    ptrs=(ctypes.c_void_p*13)(*[z.ptr for z in arrays+[p,out]])
    assert lib.inverse_run(len(params),n,nt,nd,int(two),int(entry),a["dx"],dt or (24 if n<=96 else 6 if n>=288 else 12),tau,ptrs)==0
    res=out.numpy()
    initial=arrays[1].numpy()
    np.testing.assert_allclose(res[:,0,0,:n],np.broadcast_to(initial,(len(params),n)),rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(res[:,0,0,n+NC],frame.measured_cal.to_numpy()[idx[0]],rtol=1e-12)

    if not np.isfinite(res).all() or res[:,0].min() < -1e-7:raise ValueError(("invalid trajectory",n,k,seg,entry,two,tau,params.tolist(),np.unravel_index(np.argmin(res[:,0]),res[:,0].shape),res[:,0].min()))
    density=res[:,:,:,:n+NC]@mapper.T
    return density,res[:,:,:,n+NC:]
