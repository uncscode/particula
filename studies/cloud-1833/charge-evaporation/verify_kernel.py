"""Check stochastic event probabilities against small, exactly soluble systems."""
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import expm
import warp as wp

shared, target=sys.argv[1:]
lib=ct.CDLL(shared)
lib.clusters.argtypes=[ct.c_int]*4+[ct.c_double]*2+[ct.POINTER(ct.c_void_p)]
n=160000
def check(name,q0,steps,physics,air,gamma,expected):
    start=1.
    host=[np.array([0.,2.]),np.ones(2),np.tile(np.asarray(physics,dtype=float),(4,1)),
          np.tile(np.asarray(air,dtype=float),(2,1)),gamma,np.zeros(n),
          np.zeros(n,'i4'),np.full(n,q0,'i4'),np.random.default_rng(183350).integers(1,2**63,n,dtype='uint64')]
    device=[wp.array(a,device='cuda:0') for a in host]
    out=wp.empty((n,8),dtype=wp.float64,device='cuda:0')
    hist=wp.zeros((2,2),dtype=wp.int32,device='cuda:0')
    wp.synchronize()
    pointers=(ct.c_void_p*11)(*[x.ptr for x in device+[out,hist]])
    assert lib.clusters(n,2,int(q0!=0),0,start,start+steps,pointers)==0
    z=out.numpy()
    np.testing.assert_allclose(z[:,2],start+z[:,4]-z[:,5])
    obs={'admitted':np.mean(z[:,1]==1),'background_loss':np.mean(z[:,1]==-1),
         'field_loss':np.mean(z[:,1]==-2),'neutral_alive':np.mean((z[:,1]==0)&(z[:,3]==0))}
    stats={}
    for key,p in expected.items():
        tolerance=6*np.sqrt(p*(1-p)/n)+1/n
        assert abs(obs[key]-p)<=tolerance,(name,key,obs[key],p)
        stats[key]={'simulated':obs[key],'exact':p,'tolerance':tolerance}
    return stats
g=np.zeros((4,3))
p=1-np.exp(-3)
findings={'acid_capture_competing_with_wall':check('capture',0,1,[1,.5,0,0,0,0,1,1],[0,0,0],g,
              {'admitted':2*p/3,'background_loss':p/3})}
p=(1-np.exp(-4))/2
findings['neutralization_competing_with_clearing']=check('clearing',1,1,[0,0,0,1,0,1,0,0],[1,0,1],g,
              {'neutral_alive':p,'field_loss':p})
g[1,1]=3.
p=expm(2*np.array([[-1.,1,0],[3,-4,1],[0,0,0]]))[0,2]
findings['growth_with_reversible_dimer']=check('evaporation',0,2,[1,0,0,0,0,0,1,1],[0,0,0],g,{'admitted':p})
Path(target).write_text(json.dumps({'trajectories_per_test':n,'checks':findings},indent=2)+'\n')
