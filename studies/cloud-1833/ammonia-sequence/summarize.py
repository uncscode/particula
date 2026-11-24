"""Compare every retained simulation to both archive crossings on the same clock."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
def make(root):
    rows=[]
    for f in sorted(root.glob('[0-9][0-9]-*/*/result.json')):
        run=json.loads(f.read_text())
        series=pd.read_csv(f.parent/'series.csv')
        early=series.stage.isin([9,10]);dark=(series.stage==18)
        row=dict(study=f.parents[1].name,case=f.parent.name,source_cm3=run['source_cm3'],
            reached_1000_min_error=run['errors_min'][0],reached_5000_min_error=run['errors_min'][1],
            space_min=run['rise'],max_cm3=run['peak'],flux29_stage11=run['stage11_flux29'],
            early_tail_mae=(series['tail'][early]-series.obs[early]).abs().mean(),
            mode12_nm=run['stage12_mode_nm'],small_ion_mae_cm3=run['ion_stage10_12_small_mae'],
            retained_fraction_dark=run['dark_retention'],spectral_error_factor=np.exp(np.log(run['error_factor_by_stage'][:9]).mean()),
            run_seconds=run['run_seconds'],boxes=run['boxes'],relative_sulfate_error=run['max_sulfate_residual'])
        rows.append(row)
    tab=pd.DataFrame(rows);tab.to_csv(root/'comparison.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(10,7),height_ratios=[1.25,1],layout='constrained')
    h=axs[0];base=pd.read_csv(root/'00-exchange/dynamic/series.csv')
    series=[('00-exchange/dynamic','Molecular seeds','#c06a36'),('05-neutral-embryo/paired','Bound neutral tetramers','#7a5195'),
            ('07-acid-history/all','75% effective at 1.7 nm, measured acid','#25808c'),('08-lamp-mixing/moderate','Same source, two regions','#457338'),
            ('10-signed-transfer/half_slow','Selected on signed counts','#bb3254')]
    h.plot(base.archive_h,base.obs,color='#13181f',lw=2.6,label='CLOUD measurement')
    for folder,label,color in series:
        s=pd.read_csv(root/folder/'series.csv')
        h.plot(s.archive_h,s['tail'],lw=1.6,label=label,color=color)
    for val in [1000,5000]:h.axhline(val,ls=':',color='#666666',lw=.9)
    h.set_ylim(bottom=0);h.set_xlabel('Archive time (hours)');h.set_ylabel('Particles ≥9.7 nm (cm⁻³)')
    h.legend(ncol=2,fontsize=8,frameon=False)
    a=axs[1]
    for d in rows:
        x=d['reached_1000_min_error'];y=d['reached_5000_min_error']
        if x is not None and y is not None:a.scatter(x,y,color='#617a93',alpha=.6,s=22)
    for folder,label,color in series:
        row=tab[(tab.study+'/'+tab.case)==folder].iloc[0]
        if pd.notna(row.reached_5000_min_error):a.scatter(row.reached_1000_min_error,row.reached_5000_min_error,label=label,color=color,s=52)
    a.axhline(0,color='#aaaaaa',ls=':');a.axvline(0,color='#aaaaaa',ls=':');a.scatter([0],[0],marker='*',s=100,color='black',label='Measured crossings')
    a.set_xlabel('Error at 1,000 cm⁻³ (minutes, late is positive)');a.set_ylabel('Error at 5,000 cm⁻³ (minutes)')
    a.set_title('Each point is a simulation that reached both thresholds',loc='left',fontsize=11)
    fig.savefig(root/'sequence.png',dpi=150)
if __name__=='__main__':
    import sys
    make(Path(sys.argv[1]))
