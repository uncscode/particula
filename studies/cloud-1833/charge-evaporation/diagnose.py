"""Stage-wise residuals, censored thresholds, dark decay, and independent signed data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

out=Path(sys.argv[1])
times=pd.read_csv(out/'timeseries.csv')
cases=pd.read_csv(out/'cases.csv').set_index('case')
events=pd.read_csv(out/'events.csv').set_index('case')
stage=pd.read_csv(out/'stage-spectra.csv')
signed=pd.read_csv(out/'charge-timeseries.csv')
ion=pd.read_csv(Path(__file__).parent/'ion-reference.csv')
data=pd.merge_asof(times[['archive_h']],ion.sort_values('archive_h'),on='archive_h',
                   tolerance=.001,direction='nearest')
assert not data.stage.isna().any()
assert (events.births==events[['admitted','in_transit','cluster_field_loss','cluster_background_loss','cluster_dissociation']].sum(axis=1)).all()
assert not events.max_downstream_net_charge_residual.any()
assert (cases.max_sulfate_budget_error<1e-9).all()
assert events.loc['reference',['cluster_field_loss','cluster_chargings','cluster_neutralizations','acid_evaporations']].sum()==0
assert events.loc['charge','acid_evaporations']==0 and events.loc['evaporation','cluster_chargings']==0
def first(a,v):
    hits=times.archive_h.to_numpy()[a>=v]
    return float(hits[0]) if len(hits) else np.nan
ref_stage=stage.query("case=='reference'").set_index('stage')
bootstrap=np.random.default_rng(183331).integers(0,9,(10000,9))
rows=[]
for name,values in [('measured',times.observed_tail),*[(key,times[key]) for key in cases.index]]:
    dark=data.stage==18
    row=dict(case=name,first_archive_hour=first(values,1000),second_archive_hour=first(values,5000),
             peak_over_9p7_cm3=values.max(),dark_first_cm3=values[dark].iloc[0],
             dark_last_cm3=values[dark].iloc[-1],dark_retention=values[dark].iloc[-1]/values[dark].iloc[0])
    if name!='measured':
        sub=stage[stage.case==name].set_index('stage')
        delta=np.log(sub.spectral_factor_4_to_64nm/ref_stage.spectral_factor_4_to_64nm).loc[9:17].to_numpy()
        resampled=np.exp(delta[bootstrap].mean(1))
        row.update(stage_factor_relative_to_reference=np.exp(delta.mean()),
                   stage_block_p025=np.quantile(resampled,.025),stage_block_p975=np.quantile(resampled,.975),
                   fraction_reaching_2p9=events.loc[name,'admitted']/events.loc[name,'births'])
    rows.append(row)
pd.DataFrame(rows).to_csv(out/'diagnostics.csv',index=False)
# Average predictions mapping to a given native scan, then weight that acquisition once.
bands=['ge2p2','2p2_to_2p9','2p9_to_5','5_to_9p7','9p7_to_40']
records=[]
for name,curves in signed.groupby('case',sort=False):
    paired=data.merge(curves,on=['archive_h','stage'],validate='one_to_one')
    for sid,stage_frame in paired.groupby('stage'):
        sample=stage_frame.groupby('ais_native_row').mean(numeric_only=True)
        for pol in ['negative','positive']:
            for band in bands:
                pred=sample[f"{pol}_cm3" if band=='ge2p2' else f"{pol}_{band}_cm3"]
                obs=sample[f"ais_{pol}_{band}_cm3"]
                valid=obs.notna()
                records.append(dict(case=name,stage=sid,polarity=pol,band=band,scans=valid.sum(),
                    measured_median=obs[valid].median(),predicted_median=pred[valid].median(),
                    mean_absolute_error=(pred[valid]-obs[valid]).abs().mean()))
pd.DataFrame(records).to_csv(out/'ion-diagnostics.csv',index=False)
