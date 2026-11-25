# %%
"""Independent checks of native transport, freezing counts, and phase water budgets."""
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
from particula.coupling.grid_run import GridRun
from particula.coupling.grid_inventory import InventoryLayout
from cloud import Case,geometry
import physics as g


def run(out):
    wp.init();device='cuda:0';grid,frames,sat,_,_=geometry(Case())
    # Spatially variable passive moisture with open-face boundary counts.
    vapor=np.linspace(.0009,.0017,100)[:,None]*grid.volume[:,None]
    inlet=np.full((100,6,1),.0013)
    cpu,gpu=[GridRun(frames,InventoryLayout(('water',)),vapor,inlet=inlet,device=d) for d in ('cpu',device)]
    cpu.advance();gpu.advance()
    error=float(np.max(abs(cpu.state-gpu.state.numpy())))
    np.testing.assert_allclose(cpu.state,gpu.state.numpy(),rtol=2e-14)
    np.testing.assert_allclose(cpu.imports,gpu.imports.numpy(),rtol=2e-14)
    np.testing.assert_allclose(cpu.exports,gpu.exports.numpy(),rtol=2e-14)
    # Stationary, liquid, tightly controlled test of cumulative GPU freeze hazard.
    n=20000; a=np.zeros(n,dtype=g.State.numpy_dtype())
    a['fate']=1;a['phase']=1;a['birth']=-1;a['first']=-1
    a['rate']=.4;a['mass']=4*np.pi/3*1000*(9e-6)**3;a['weight']=1;a['dry_radius']=65e-9
    u=np.random.default_rng(9104).exponential(size=n);a['threshold']=u
    arr=lambda x:wp.array(np.ascontiguousarray(x),dtype=wp.float64,device=device)
    salt=arr(np.array([[.0016,.0014,5e-11,5e-11,-14.]])); scale=arr(np.ones(1))
    s=wp.array(a,dtype=g.State,device=device);clk=arr(np.zeros(1))
    for _ in range(120):
        wp.launch(g.phase_step,n,[s,scale,salt,clk,60.,0,0.,1.,0,123],device=device)
        wp.launch(g.tick,1,[clk,60.],device=device)
    final=s.numpy();assert np.array_equal(final['freezes']==1,u<=.8)
    freq=final['freezes'].mean();expect=-np.expm1(-.8)
    # Cross both phase boundaries with the same three condensation nuclei, at zero rate.
    a=a[:3].copy();a['rate']=0;a['phase']=0;a['mass']=0;a['wets']=0
    s=wp.array(a,dtype=g.State,device=device);total=arr(np.zeros(1));scale=arr(np.ones(1))
    used=[];errs=[]
    for vapor0 in [.00162,.0010,.00162]:
        v=arr(np.array([[vapor0]]));before=vapor0+s.numpy()['mass'].sum()
        wp.launch(g.propose,3,[s,v,salt,1.,60.,clk,0,total],device=device)
        wp.launch(g.limit,1,[v,salt,total,scale,1.,0],device=device)
        wp.launch(g.phase_step,3,[s,scale,salt,clk,60.,0,0.,1.,0,123],device=device)
        st=s.numpy();used.append(int(st['phase'][0]));errs.append(abs(v.numpy().item()+st['mass'].sum()-before))
    assert used==[1,0,1] and np.max(errs)<1e-16 and (st['wets']==2).all() and (st['dries']==1).all()
    result=dict(transport_max_abs_kg=error,constant_hazard_particles=n,expected_fraction=expect,
                gpu_fraction=freq,threshold_id_matches=n,phase_cycle=used,cycle_max_abs_kg=max(errs))
    (out/'oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__':
    run(Path(sys.argv[1]))
