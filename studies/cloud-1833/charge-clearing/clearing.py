"""Single clearing pulse applied to an otherwise identical archived one-zone replay."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warp as wp
from physics import PACKET, d, e, geometry, segments, stage, t, y
idx=segments["test"]
hours=t[idx]
switch=np.flatnonzero(stage[idx]==11)[0]
gas,entry=np.load(PACKET/"inverse-model/final-gas.npz")["test_288_1_1_0_80"][:,:2].T
params=pd.read_csv(PACKET/"inverse-model/frozen-final.csv").query("k==1 and entry==1").iloc[0]
obs=y[idx[0]].copy()
obs[3]=np.interp(np.log(d[3]),np.log(d[[2,4]]),obs[[2,4]])
tail=np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))
def crossings(curve):
    return [hours[np.flatnonzero(curve>=v)[0]] for v in [1000,5000]]
ref=crossings(y[idx]@tail)
sol=ct.CDLL(os.environ['CLOUD_SOLVER'])
sol.replay.argtypes=[ct.c_int]*3+[ct.c_double]*2+[ct.POINTER(ct.c_void_p)]
records=[]
for n in (192,288):
    a=geometry(n,'aiomfac',1.)
    hard,attractive=a['vel']
    v=np.repeat((hard+params.collision_fraction*(attractive-hard))[None,:],3,axis=0)
    initial=a['mi']@obs
    for fraction in [0.,.2,1.]:
        clear=np.zeros_like(hours);clear[switch]=fraction
        inputs=[hours,gas,np.ones_like(hours),initial,np.r_[v.ravel(),v.max(1)],a['wall'],
                np.array([1.1,np.exp(params.log_particle_wall),1.,1.]),a['K'],*a['pairs'],a['over'],clear,entry]
        gpu=[wp.array(np.ascontiguousarray(x),device='cuda:0') for x in inputs]
        state=wp.empty((1,len(hours),n),dtype=wp.float64,device='cuda:0')
        ledger=wp.empty((1,n,6),dtype=wp.float64,device='cuda:0')
        pointers=(ct.c_void_p*17)(*[v.ptr for v in gpu+[state,ledger]])
        wp.synchronize()
        assert sol.replay(1,n,len(hours),a['dx'],6 if n==288 else 12,pointers)==0
        x=state.numpy()[0];b=ledger.numpy().sum(axis=(0,1))
        np.testing.assert_allclose(x[0],initial)
        assert np.isfinite(x).all() and x.min()>=0
        residual=abs(x[-1].sum()-initial.sum()-b[0]+b[1:].sum())/(initial.sum()+b[0])
        assert residual<1e-9
        curve=(x@a['mo'].T/np.diff(e))@tail
        first,second=crossings(curve)
        records.append(dict(cells=n,pulse_fraction=fraction,field_archive_hour=hours[switch],removed_cm3=b[-1],
                   first_delay_min=60*(first-ref[0]),second_delay_min=60*(second-ref[1]),rise_min=60*(second-first),
                   number_balance=residual))
df=pd.DataFrame(records)
df.to_csv(Path(os.environ['RESULTS'])/'clearing.csv',index=False)
print(df.round(3).to_string(index=False))
