"""An instrumented replay reads the exact first passages across 5 nm."""
import ctypes as ct, json, os, runpy, sys
from pathlib import Path
import numpy as np
import pandas as pd
packet=Path(os.environ['PACKET'])
parent=packet/'ammonia-sequence'
sys.path.insert(0,str(parent))
libdir,output,temp=map(Path,sys.argv[1:4])
settings=parent/'10-signed-transfer/half_slow.json'
sys.argv=[str(parent/'simulate.py'),str(settings),str(temp),str(libdir)]
res=runpy.run_path(sys.argv[0],run_name='__main__')
nt=len(res['ch'].hours);boxes=res['boxes']
counts=np.zeros((boxes,nt-1,2),dtype='i4')
lib=ct.CDLL(str(libdir/'particles.so'));lib.read_crossings.argtypes=[ct.c_void_p]
lib.read_crossings(counts.ctypes.data)
h=res['ch'].hours;dt=np.diff(h)*3600
pop=counts.sum(0)/(res['vol']*boxes*dt[:,None])
pd.DataFrame(dict(start_h=h[:-1],end_h=h[1:],stage=res['ch'].st[:-1],growth_cm3_s=pop[:,0],
    merger_cm3_s=pop[:,1],total_cm3_s=pop.sum(1))).to_csv(output/'passages.csv',index=False)
for name in ['signed.csv','series.csv']:
    pd.testing.assert_frame_equal(pd.read_csv(temp/name),pd.read_csv(parent/'10-signed-transfer/half_slow'/name),check_exact=True)
(output/'passage-check.json').write_text(json.dumps(dict(case='10-signed-transfer/half_slow',boxes=boxes,volume_cm3=res['vol'],
    uptake_crossings=int(counts[:,:,0].sum()),merger_crossings=int(counts[:,:,1].sum()),
    archived_count_series_and_signed_diagnostics='exactly reproduced',seconds=json.loads((temp/'result.json').read_text())['run_seconds']),indent=2))
