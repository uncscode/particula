"""Compute loss of resident freezing and incoming yield per admitted carrier."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from campaign import DUST,SAMPLES,BASE_SEED
from summarize import lookup

def intervals(x):
    n=len(x)
    if n<1:return {}
    half=x.sem()*t.ppf(.975,n-1) if n>1 else np.nan
    return dict(n=n,mean=x.mean(),lo=x.mean()-half,hi=x.mean()+half)

def write(d,out):
    rows=[]
    for (sky,grid,abundance),g in d.groupby(['sky','grid','abundance']):
        common=dict(sky=sky,grid=grid,abundance=abundance)
        closed=lookup(g,**common,donor='closed').set_index('seed')
        if closed.empty:continue
        for source in SAMPLES:
            x=lookup(g,**common,donor=source).set_index('seed')
            paired=x.index.intersection(closed.index)
            spaced=paired[paired>=BASE_SEED+10000019]
            if len(spaced):paired=spaced
            if not len(paired):continue
            x=x.loc[paired]
            replacement=closed.loc[paired,'resident_primary']-x.resident_primary
            imports=x.fresh_inflow_or_initial
            values={'resident_loss_L':replacement,'incoming_L':x.fresh_primary,'new_minus_closed_L':x.fresh_primary-replacement,
                    'net_per_100_admitted':100*(x.fresh_primary-replacement)/imports,
                    'required_per_100_admitted':100*replacement/imports,
                    'produced_per_100_admitted':100*x.fresh_primary/imports,
                    'hours_wet_per_admitted':x.incoming_wet_h/imports,'ever_wet_per_100_admitted':100*x.incoming_ever_wet/imports}
            np.testing.assert_allclose(values['new_minus_closed_L'],x.primary-closed.loc[paired,'primary'],atol=1e-12)
            for metric,result in values.items():rows.append(dict(sky=sky,grid=grid,abundance=abundance,source=source,metric=metric,**intervals(result)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'resident-balance.csv',index=False)
    return frame
