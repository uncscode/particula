"""Publish compact endpoint evidence for comparisons discussed in the cloud review."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from campaign import DUST, SAMPLES
from summarize import load, lookup, paired, EXPERIMENT

BASE=['closed','aged_'+DUST,DUST,'RuralContinental','Marine']
METRICS=['primary','unique','suspended','LWP','IWP','resident_primary','fresh_primary','incoming_wet_h',
         'fresh_live_ice','resident_recycled_primary','fresh_recycled_primary','fresh_inflow_or_initial',
         'incoming_ever_wet','resident_wet_h']


def report(d,out):
    d=d.drop_duplicates(EXPERIMENT)
    from budget_figure import write
    write(d,out)
    from interactions import write as write_interactions
    write_interactions(d,out)
    rows=[]
    def compare(test, sky, name, control_sky, control_name, common=None, left=None, right=None, label=''):
        opt=common or {}
        a=lookup(d,**{**opt,'sky':sky,'donor':name,**(left or {})})
        b=lookup(d,**{**opt,'sky':control_sky,'donor':control_name,**(right or {})})
        if a.empty or b.empty:return
        for metric in METRICS:
            result=paired(a,b,metric)
            if result:rows.append(dict(test=test,sky=sky,source=name,comparison=control_name,
                                       control_sky=control_sky,setting=label,metric=metric,**result))
    for sky in ['reference','vigorous_dry','isdac','mpace','cold_deep','warm_thin','weak_dry','moist_mixed',
                'isdac_we4p125','mpace_we8p2']:
        for source in SAMPLES:
            for grid in [10,20,30,40,60,80]:
                for abund in ['equal','air_volume']:
                    if len(lookup(d,sky=sky,donor=source,grid=grid,abundance=abund))==0:continue
                    for ref in ['closed','aged_'+DUST]:
                        compare('core',sky,source,sky,ref,dict(grid=grid,abundance=abund),label=f'{abund}-{grid}')
        for abundance in ['equal','air_volume']:
            for grid in [10,20,30,40,60]:
                compare('marine_minus_dust',sky,'Marine',sky,DUST,common=dict(abundance=abundance,grid=grid),label=f'{abundance}-{grid}')
        for src in SAMPLES:
            compare('source_age',sky,src,sky,'aged_'+src)
    for sky in ['reference','isdac','mpace','vigorous_dry','mix2','mix8','synthetic_roll400_1800']:
        for grid,density,step in [(10,48,1.),(10,48,.5),(10,48,.25),(10,96,1.),(20,48,1.),(30,48,1.),(40,48,1.),(60,48,1.)]:
            for src in [DUST,'ContinentalPristine','RuralContinental']:
                for ref in ['aged_'+DUST,'closed']:
                    compare('numerics',sky,src,sky,ref,common=dict(grid=grid,density=density,dt_factor=step),
                            label=f'grid{grid}_density{density}_dt{step}')
    for sky in ['reference','vigorous_dry']:
        for donor in [DUST,'Marine','RuralContinental']:
            for share in [0.,.25,.5,.75,1.]:
                compare('active_fraction',sky,donor,sky,'closed',left=dict(share=share),label=str(share))
    for sky in ['isdac','reference']:
        for abundance in ['equal','air_volume']:
            for resident in [DUST,'Marine','RuralContinental']:
                for source in [DUST,'Marine']:
                    for control in ['closed','aged_'+resident]:
                        compare('resident_spectrum',sky,source,sky,control,
                                common=dict(abundance=abundance,resident=resident),label=f'{resident}-{abundance}')
    for sky in ['reference','vigorous_dry']:
        for src in SAMPLES:
            for pl in range(1,6):
                compare('plate',sky,src,sky,'aged_'+DUST,left=dict(plate=pl),label=str(pl))
    for sky in ['reference','cold_deep','isdac','warm_thin']:
        for src in ['ContinentalPolluted',DUST,'Marine']:
            for slope in [0.,.25,.5,.75]:
                compare('thermal',sky,src,sky,'aged_'+DUST,dict(slope=slope),label=str(slope))
    for base,target in [('reference','mix05'),('reference','mix2'),('reference','mix8'),
                        ('mix2','synthetic_roll0_1800'),('mix2','synthetic_roll200_1800'),
                        ('mix2','synthetic_roll400_1800'),('mix2','synthetic_roll200_600'),
                        ('isdac','isdac_roll200_1800'),('mpace','mpace_roll200_1800'),
                        ('qbase','qplus025'),('qbase','qplus050'),
                        *[('mpace','mpace_flux'+str(f)) for f in [5,10,25,50,100]],
                        ('isdac','isdac_ccn50'),('mpace','mpace_ccn200')]:
        for src in BASE:
            compare('forcing',target,src,base,src)
            compare('forcing_net',target,src,target,'closed')
    for sky in ['reference','cold_deep','very_cold']:
        for src in ['grain_feldspar','grain_quartz']:
            for diam,num,shift in [(1.,1000.,0.),(.5,1000.,0.),(2.,1000.,0.),(1.,100.,0.),
                                   (1.,10.,0.),(1.,1000.,-.8),(1.,1000.,.8)]:
                compare('minerals',sky,src,sky,'grain_inert',common=dict(grain_number=num,grain_diameter=diam),
                        left=dict(site_shift=shift),label=f'{diam}-{num}-{shift}')
    for sky in ['reference','mix2','very_cold','cold_deep']:
        for material in ['feldspar','quartz']:
            compare('matched_mineral_sham',sky,'grain_'+material,sky,'grain_sham_'+material)
    compare('direct_sampling','very_cold','grain_direct_feldspar','very_cold','grain_feldspar',
            common=dict(grain_diameter=2.))
    table=pd.DataFrame(rows)
    table.to_csv(out/'evidence.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('ledger',type=Path);parser.add_argument('out',type=Path)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True);report(load(a.ledger),a.out)
