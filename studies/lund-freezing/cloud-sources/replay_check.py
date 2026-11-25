"""Compare an independently regenerated subset with registered production trajectories."""
import argparse
from pathlib import Path
import gzip
import json
import numpy as np
from campaign import Experiment, key


def read(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:
        return {key(Experiment(**r['experiment'])):r for line in f if line.endswith('\n') for r in [json.loads(line)]}


def verify(original,replay):
    a,b=read(original),read(replay)
    assert len(b)>=6
    for name,y in b.items():
        x=a[name]
        for section in ['values','diagnostics']:
            for k in x[section]:
                if k=='mean_LWP' and k not in y[section]:continue
                value=x[section][k]
                if isinstance(value,str):
                    assert value==y[section][k]
                else:
                    np.testing.assert_allclose(value,y[section][k],rtol=3e-10,atol=2e-10,err_msg=k)
        assert len(x['evolution'])==len(y['evolution'])
        for s,t in zip(x['evolution'],y['evolution']):
            np.testing.assert_allclose(list(s.values()),list(t.values()),rtol=3e-10,atol=2e-10)
        assert x['integrator']['GPU']==y['integrator']['GPU']
    return len(b)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('production',type=Path);ap.add_argument('replay',type=Path)
    args=ap.parse_args()
    print(verify(args.production,args.replay),'cleanly replayed cases match source outcomes, water budgets and saved time histories.')
