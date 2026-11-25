# %%
"""Predict Lund isothermal freezing with a population law and persistent rates."""
import argparse
import hashlib
import json
import re
import tarfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SAMPLES = ["Marine", "MineralDust", "ContinentalPristine", "ContinentalPolluted",
           "CombustionDominated", "RuralContinental"]
EDGES = np.array([0,15,30,60,120,300,600,1200,1800,3600,5400,7200])/3600
HOURS = np.array([0, .5, 2, 4, 6, 8, 10])
HASHES = ["cbc85471842794a0f3c16c9c58f7a8b05a5507b066871316d77bd7ac01ba9eb7",
          "16a664ab91ab5b7b173116d8c748e796e0a5af7c6a8dfc09adf30f74f62bf790",
          "eb5127e695ff2e77143b587916f9cecf58a9189b472b60676f29fcdd5355397c"]


def tables(path, start):
    with ZipFile(path) as z:
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    return np.array([[float(c.find('{*}v').text) for c in row.findall('{*}c')[1:]]
                     for row in sheet.findall('.//{*}row')[start:]])


def read_inputs(path, pilot):
    files = [path / f"lund-study-{name}" for name in ("events.tar","table3.xlsx","table4.xlsx")]
    for f, expected in zip(files, HASHES, strict=True):
        assert hashlib.sha256(f.read_bytes()).hexdigest() == expected
    data = []
    with tarfile.open(files[0]) as archive:
        for m in archive:
            if 'DropPopulation' not in m.name:
                continue
            name = Path(m.name).stem
            sample, typ, duration = name.split('_')[:3]
            phase = 'ramp' if typ == 'ConstantCoolingRate' else ('short' if duration == '120min' else 'night')
            if pilot and (sample != 'Marine' or phase == 'night'):
                continue
            arr = np.loadtxt(archive.extractfile(m))
            assert np.array_equal(np.sort(arr[:,2]), np.arange(1,101))
            frame = pd.DataFrame(arr,columns=['temp','seconds','id','x','y'])
            frame['sample'], frame['phase'], frame['run'] = sample,phase,name
            frame['plate'] = int(re.search('DropPopulation_(..)', name)[1])
            target = -16 if sample == 'ContinentalPolluted' else -14
            frame['target'] = target
            frame['h'] = (frame.seconds - (360 if target == -16 else 300))/3600
            if phase != 'ramp':
                last = 2 if phase == 'short' else 10
                assert ((frame.h<=0)|(frame.h>last)|(frame.temp==target)).all(), name
            data.append(frame)
    data = pd.concat(data,ignore_index=True)
    data.id = data.id.astype(int)
    return data, tables(files[1],3), tables(files[2],2)


def geometry(data):
    keys = ['sample','plate','id']
    anchor = data[data.phase=='short'].groupby(keys)[['x','y']].median()
    data = data.join(anchor, on=keys,rsuffix='0')
    data['shift'] = np.hypot(data.x-data.x0,data.y-data.y0)
    for axis in ['x','y']:
        data[f'{axis}c'] = data.groupby(['sample','plate','run'])[axis].transform('median')
    data['radius'] = np.hypot(data.x-data.xc,data.y-data.yc)
    data['inner'] = data.radius <= data.groupby(['sample','plate','run']).radius.transform('median')
    return data


def counts(times):
    # Conditional on hold-entry survival. Events exactly on the upper edge belong to that bin.
    pos = np.searchsorted(EDGES[1:],times,side='left')
    return np.bincount(pos,minlength=len(EDGES))


def weibull(t,x):
    fraction,scale,power = x
    return fraction*(-np.expm1(-(np.asarray(t)/scale)**power))


def fit_population(short):
    obs = counts(short.h.to_numpy())
    def nll(x):
        cdf = weibull(EDGES,x)
        probs = np.r_[np.diff(cdf),1-cdf[-1]]
        return -obs @ np.log(np.maximum(probs,1e-100))
    fits = [minimize(nll, initial, method='Nelder-Mead',
                     bounds=[(.001,1),(.0005,500),(.06,2)],
                     options={'maxiter':2500,'xatol':1e-9,'fatol':1e-8})
            for initial in ([.35,.8,.6],[.9,4,.4],[.2,.15,.9])]
    return min(fits,key=lambda fit: fit.fun).x


def prior_fit(likelihood, center=None, strength=1, steps=180):
    # Mix over possible fixed per-well rates; shrink sparsely sampled strata.
    if center is None:
        center = np.ones(likelihood.shape[1])/likelihood.shape[1]
    weight = center.copy()
    for _ in range(steps):
        post = weight*likelihood
        post /= post.sum(axis=1,keepdims=True)
        weight = (post.sum(axis=0)+strength*center)/(len(post)+strength)
    return weight


def survival_law(hours,rates,power=1.0,gamma=0.0):
    if gamma:
        return np.exp(-gamma*np.log1p(np.outer(hours,rates)/gamma))
    return np.exp(-np.outer(np.asarray(hours)**power,rates))


def train(sample, nrate=21, floor=0, ramp=True, match=80, power=1.0, gamma=0.0, steps=180, strength=16):
    short = sample[(sample.phase=='short') & (sample.h>0)]
    ref = fit_population(short)
    # h^-1; very fast rates allow freezing during the first binned frame.
    rates = np.r_[floor,np.geomspace(max(.002,floor*1.1),128,nrate)]
    s = survival_law(EDGES,rates,power,gamma)
    prob = np.vstack([-np.diff(s,axis=0),s[-1]])
    keys = pd.MultiIndex.from_product([range(1,6),range(1,101)],names=['plate','id'])
    n = np.zeros((500,len(EDGES)))
    for key, group in short[short['shift']<=match].groupby(['plate','id']):
        n[keys.get_loc(key)] = counts(group.h)
    ll = n @ np.log(np.maximum(prob,1e-120))
    like = np.exp(ll-ll.max(axis=1,keepdims=True))
    prior = prior_fit(like[n.sum(axis=1)>0],steps=steps)
    temp = sample[(sample.phase=='ramp') & (sample['shift']<=match)].groupby(['plate','id']).temp.median().reindex(keys).to_numpy()
    target = sample.target.iloc[0]
    strata = np.floor(np.nan_to_num(temp-target,nan=-99)).clip(-8,4) if ramp else np.zeros(500)
    weights = np.tile(prior,(500,1))
    for k in np.unique(strata):
        use = (strata==k)&(n.sum(axis=1)>0)
        if use.any():
            weights[strata==k] = prior_fit(like[use],prior,strength,steps=steps)
    posterior = weights*like
    posterior /= posterior.sum(axis=1,keepdims=True)
    return ref,rates,prior,weights,posterior


def predictions(sample, trained, match=80, power=1.0, gamma=0.0):
    ref,rates,prior,weights,posterior = trained
    night = sample[(sample.phase=='night') & (sample.h>0)].copy()
    ids = ((night.plate-1)*100+night.id-1).to_numpy()
    survival = survival_law(HOURS,rates,power,gamma).T
    for label,source in [('persistent',posterior),('ramp_only',weights)]:
        w = source[ids].copy()
        w[night['shift']>match] = prior
        for j,t in enumerate(HOURS):
            night[f'{label}_{t:g}'] = 1-w @ survival[:,j]
    for t in HOURS:
        night[f'population_{t:g}'] = weibull(t,ref)
        night[f'observed_{t:g}'] = (night.h<=t).astype(int)
    return night


def published_reference(data,t3,t4):
    rows = []
    for i,sample in enumerate(SAMPLES):
        d = data[(data['sample']==sample)&(data.phase!='ramp')]
        for t in np.unique(np.r_[HOURS,np.geomspace(10/3600,10,60)]):
            use = d if t<=2 else d[d.phase=='night']
            f0,d2,d10 = t3[i,:3]
            c,alpha=t4[i,[0,3]]
            obs = (use.h<=t).mean()
            rows.append(dict(sample=sample,h=t,raw=obs,
                             short_only=(d[d.phase=='short'].h<=t).mean() if t<=2 else np.nan,
                             overnight=(d[d.phase=='night'].h<=t).mean(),
                             paper=f0+d10*(-np.expm1(-(3600*t)**(1-alpha)/c)),
                             table={0:f0,2:f0+d2,10:f0+d10}.get(t,np.nan)))
    return pd.DataFrame(rows)


def score(d, variant):
    rows = []
    for (name,plate),g in d.groupby(['sample','plate']):
        for subset,selected in [('all',g),('inner',g[g.inner]),('matched',g[g['shift']<=80])]:
            for model in ['population','persistent','ramp_only']:
                p=lambda t:selected[f'{model}_{t}'].to_numpy()
                late = selected.h.to_numpy()>2
                early_event = (selected.h.to_numpy()<=.5).astype(float)
                late_event = (selected.h.to_numpy()[late]<=10).astype(float)
                cond = (p(10)[late]-p(2)[late])/(1-p(2)[late])
                actual = 100*np.mean((selected.h>2)&(selected.h<=10))
                predicted = 100*np.mean(p(10)-p(2))
                rows.append(dict(sample=name,plate=plate,variant=variant,subset=subset,model=model,
                                 n=len(selected),n2=late.sum(),late=actual,predicted=predicted,
                                 error=predicted-actual,abs_error=abs(predicted-actual),
                                 brier_early=np.mean((p(0.5)-early_event)**2),
                                 id_gain_early=np.mean((p(0.5).mean()-early_event)**2-(p(0.5)-early_event)**2),
                                 brier_late=np.mean((cond-late_event)**2),
                                 id_gain_late=np.mean((cond.mean()-late_event)**2-(cond-late_event)**2),
                                 conditional_late=100*cond.mean(),observed_conditional=100*late_event.mean(),
                                 minus60=100*np.mean((selected.h>2+1/60)&(selected.h<=10-1/60)),
                                 plus60=100*np.mean((selected.h>2-1/60)&(selected.h<=10+1/60))))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path); parser.add_argument('output',type=Path)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=True)
    d,t3,t4=read_inputs(args.input,args.pilot)
    d=geometry(d)
    if args.pilot:
        # Reserve the lexically last short file on every plate for an internal pilot.
        runs=d[d.phase=='short'].groupby('plate').run.max()
        selected=d.run.isin(runs)
        d.loc[selected,'phase']='night'
        d=geometry(d.drop(columns=['x0','y0','xc','yc','shift','radius','inner']))
        t=train(d)
        pp=predictions(d,t)
        metrics=score(pp,'pilot')
        pd.DataFrame(metrics).query("subset=='all'")[["sample","plate","model","n","brier_early","id_gain_early"]].to_csv(out/'pilot.csv',index=False)
        print('Pilot only: early Brier; late times excluded from interpretation')
        print(pd.DataFrame(metrics).query("subset=='all'").groupby('model').brier_early.mean())
        return
    all_scores=[]; all_forecasts=[]; fitted=[]
    for variant,params in [('main',{}),('fine',{'nrate':41}),('slow_floor',{'floor':.025}),
                           ('no_ramps',{'ramp':False}),('tight_match',{'match':40})]:
        for name,g in d.groupby('sample'):
            training=train(g,**params)
            pred=predictions(g,training,params.get('match',80))
            all_scores.extend(score(pred,variant))
            if variant=='main':
                all_forecasts.append(pred)
                fitted.append(dict(sample=name,fraction=training[0][0],scale_h=training[0][1],power=training[0][2]))
    pd.concat(all_forecasts).to_csv(out/'well_forecasts.csv.gz',index=False,
                               compression={'method':'gzip','mtime':0})
    pd.DataFrame(all_scores).to_csv(out/'plate_scores.csv',index=False)
    pd.DataFrame(fitted).to_csv(out/'population_fits.csv',index=False)
    published_reference(d,t3,t4).to_csv(out/'paper_reference.csv',index=False)
    print(pd.DataFrame(all_scores).query("variant=='main' and subset=='all'").groupby('model')[['late','predicted','abs_error','id_gain_early','id_gain_late']].mean())
    print(pd.DataFrame(fitted))

if __name__=='__main__':
    main()
