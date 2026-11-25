"""Verify source weights, inversion structure, mineral tables and headline accounting."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import DUST, BASE_SEED, Experiment, spectra_builder
import isdac
import mpace
from minerals import site_density


def check(folder):
    spectra=pd.read_csv(folder/'spectra.csv')
    old=pd.read_csv(Path(__file__).resolve().parents[1]/'cloud-eddy/results/rates.csv')
    dust=spectra.query("source=='MineralDust'").sort_values(['law','plate','rate'])
    old=old.sort_values(['law','plate','rate'])
    np.testing.assert_allclose(dust[['rate','fresh','aged']],old[['rate','fresh','aged']],rtol=3e-11,atol=1e-14)
    for (source,law,plate),g in spectra.groupby(['source','law','plate']):
        assert np.isclose(g.fresh.sum(),1) and np.isclose(g.aged.sum(),1)
        raw=g.fresh*np.exp(-g.rate*(2**(.6 if law=='aging' else 1.)))
        np.testing.assert_allclose(raw/raw.sum(),g.aged,rtol=1e-12,atol=2e-15)
    compose=spectra_builder(spectra)
    e=Experiment('reference','ContinentalPolluted',BASE_SEED)
    z=compose(e)
    raw=spectra.query("source=='ContinentalPolluted' and law=='fixed'").groupby('rate').fresh.mean()
    np.testing.assert_allclose(np.sum(z.fresh*(-np.expm1(-2*z.rate*np.exp(1.)))),
                               np.sum(raw*(-np.expm1(-2*raw.index))),rtol=1e-12)
    # Independent checks against the published equations and saturation-adjusted profiles.
    rows=[]
    for label,module,base_m,top_m,base_expected,lwp_expected,inlet in [
        ('ISDAC',isdac,225,825,649,14.4,1.4556),('M-PACE',mpace,732,1332,687,207.6,.9270)]:
        table=module.sounding()
        z=np.arange(len(table))
        rho_v, liquid=table[:,2],table[:,3]
        np.testing.assert_allclose(table[:,1]/((isdac.RD+(isdac.RV*(rho_v/((table[:,4])/(1+table[:,6])))))*table[:,0])*
                                   table[:,6],rho_v+liquid,rtol=1e-12)
        assert abs(z[liquid>0][0]-base_expected)<=1
        assert abs(liquid.sum()*1000-lwp_expected)<.6
        assert abs(module.inlet()*1000-inlet)<1e-4
        for alt,r in enumerate(table):
            rows.append(dict(profile=label,asl_m=alt,model_z_m=alt-base_m,temperature_C=r[0]-273.15,
                             pressure_Pa=r[1],vapor_g_m3=1000*r[2],liquid_g_m3=1000*r[3],
                             theta_l_K=r[5],total_water_g_kg=1000*r[6]))
    d=pd.DataFrame(rows); d.to_csv(folder/'profiles.csv',index=False)
    np.testing.assert_allclose(site_density('quartz',[-15.]),10**.28075,rtol=1e-12)
    np.testing.assert_allclose(site_density('feldspar',[-15.]),10**2.5451375,rtol=1e-12)
    ledger=folder/'runs.jsonl'
    opener=open
    if not ledger.exists(): ledger=ledger.with_suffix('.jsonl.gz');opener=gzip.open
    with opener(ledger,'rt') as f:runs=[json.loads(line) for line in f if line.endswith('\n')]
    original=pd.read_csv(Path(__file__).resolve().parents[1]/'cloud-eddy/results/main_history.csv').query('t_h==6')
    rollref=pd.read_csv(Path(__file__).resolve().parents[1]/'cloud-eddy/results/environment_history.csv').query('t_h==6')
    matches=0;steady=0
    for r in runs:
        e=Experiment(**r['experiment'])
        if e.sky=='reference' and e.seed<BASE_SEED+8 and e.donor in [DUST,'aged_'+DUST] and e==Experiment('reference',e.donor,e.seed):
            label='fresh' if e.donor==DUST else 'aged'
            ref=original[(original['case']==label)&(original.seed==e.seed)].iloc[0]
            metrics=['primary','LWP','IWP','resident_primary','fresh_primary','suspended']
            np.testing.assert_allclose([r['values'][k] for k in metrics],ref[metrics].astype(float),rtol=3e-10,atol=1e-10)
            matches+=1
        if e.sky=='synthetic_roll0_1800' and e.seed<BASE_SEED+4 and e.donor in [DUST,'aged_'+DUST] and e==Experiment('synthetic_roll0_1800',e.donor,e.seed):
            name='mixing_fresh' if e.donor==DUST else 'mixing_aged'
            ref=rollref[(rollref['case']==name)&(rollref.seed==e.seed)].iloc[0]
            for metric in ['primary','suspended','LWP']:
                np.testing.assert_allclose(r['values'][metric],ref[metric],rtol=3e-10,atol=1e-10)
            steady+=1
        d=r['values'];diag=r['diagnostics']
        np.testing.assert_allclose(d['resident_primary']+d['fresh_primary'],d['primary'],atol=1e-12,rtol=2e-12)
        assert d['primary']+1e-12>=d['unique']>=d['suspended']-1e-10
        assert r['integrator']['max_water_error']<1e-10
        if 'inlet_carriers_l' in diag and r['experiment']['donor']!='closed':
            expected=diag['inlet_carriers_l']*r['sky']['exchange']*r['experiment']['hours']*3600/(np.pi*600)
            np.testing.assert_allclose(d['fresh_inflow_or_initial'],expected,rtol=1e-11)
        if 'max_site_probability' in diag:
            assert d['unique']<=d['resident_inflow_or_initial']+diag['admitted_grains']*diag['max_site_probability']+1e-10 or 'direct_' in r['experiment']['donor']
        if r['sky'].get('delivered_flux',0.):
            target=1600*r['sky']['delivered_flux']*r['experiment']['hours']*3600
            np.testing.assert_allclose(d['surface_water_kg'],target,rtol=5e-12)
    assert matches==16 and steady==8,(matches,steady)
    print(f'{len(runs)} integrations checked. All 16 original fresh/aged dust baselines and eight zero-displacement controls reproduced. Source spectra, assay reference temperature, two sounding initializations, mineral equations, cumulative events, unique cores, water supply and closed water budgets passed.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);check(ap.parse_args().folder)
