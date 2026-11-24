"""Reference the individual-particle outcomes to the archived frozen sectional fit."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from physics import PACKET,segments,stage
out=Path(sys.argv[1]);tr=pd.read_csv(PACKET/'inverse-model/outcomes.csv')
r=tr[tr.key=='test_288_1_1_0_80'].iloc[0]
ts=pd.read_csv(out/'timeseries.csv');st=stage[segments['test']]
c=pd.read_csv(out/'cases.csv').set_index('case')
row=c.loc['discrete']
co=pd.read_csv(out/'cohorts.csv')
co=co[(co.case=='discrete')&(co.pathway=='all')].set_index('stage')
def rate(ss):
    hours=ts.archive_h.to_numpy()
    bounds=(hours[:-1]+hours[1:])/2
    # Stage boundaries halfway between the samples on either side of a transition.
    i=np.flatnonzero(st==ss)
    duration=(bounds[i[-1]]-bounds[i[0]-1])*3600
    return float(co.loc[ss,'births_per_cm3']/duration)
result=dict(sectional_first_delay_min=r.delay1000_min,sectional_second_delay_min=r.delay5000_min,
    sectional_rise_minutes=98+r.delay5000_min-r.delay1000_min,sectional_stage12_mode_nm=r.stage12_peak_nm,
    observed_stage12_mode_nm=9.17,particle_resolved_mc_interval_type='95% bootstrap of whole independent boxes',
    modeled_birth_rates_cm3_s={s:rate(s) for s in [10,11,14]},
    stage11_sectional_mean_entry_cm3_s=float(ts.loc[st==11,'sectional_inlet'].mean()),
    stage11_particle_resolved_mean_entry_cm3_s=row.stage11_influx_cm3_s)
(out/'benchmark.json').write_text(json.dumps(result,indent=2))
