"""Score fixed kinetic hypotheses on stage 10 and transfer frozen choices to stage 12."""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
root=Path(sys.argv[1])
last=root/'10-signed-transfer'
obs=pd.read_csv(Path(__file__).resolve().parents[1]/'charge-evaporation/ion-reference.csv')
choices={'early_normal':'08-lamp-mixing/moderate','early_slow':'09-ion-turnover/neutralization'}
for n in ['half_mature','all_mature','half_slow','all_slow']:
    choices[n]='10-signed-transfer/'+n
def errors(folder,stage):
    sim=pd.read_csv(root/folder/'series.csv')
    sim=sim.merge(obs,on=['archive_h','stage'],validate='one_to_one')
    d=sim[sim.stage==stage].groupby('ais_native_row').mean(numeric_only=True)
    cols=[f'{pol}_{band}' for pol in ('negative','positive') for band in ('2p2_to_2p9','2p9_to_5')]
    return np.abs(d[cols].to_numpy()-d[['ais_'+k+'_cm3' for k in cols]].to_numpy()).mean(1)
d={}
for stage in [10,12]:
    frame=pd.DataFrame({key:errors(folder,stage) for key,folder in choices.items()})
    d[stage]=frame
winner=d[10].mean().idxmin()
rows=[]
for sid,frame in d.items():
    rng=np.random.default_rng(183310+sid)
    count=len(frame)
    starts=rng.integers(0,count,(3000,int(np.ceil(count/4))))
    ids=((starts[:,:,None]+np.arange(4))%count).reshape(len(starts),-1)[:,:count]
    scores=frame.to_numpy()[ids].mean(1)
    for j,key in enumerate(choices):
        delta=scores[:,j]-scores[:,frame.columns.get_loc(winner)]
        rows.append(dict(stage=sid,candidate=key,folder=choices[key],mae=frame[key].mean(),
            delta_vs_selected=delta.mean(),delta_p025=np.quantile(delta,.025),delta_p975=np.quantile(delta,.975),
            bootstrap_top_fraction=np.mean(scores.argmin(1)==j),scans=count,
            selected_on_stage10=key==winner))
pd.DataFrame(rows).to_csv(last/'transfer.csv',index=False)
print(pd.DataFrame(rows).pivot(index='candidate',columns='stage',values='mae').sort_values(10).round(1))
