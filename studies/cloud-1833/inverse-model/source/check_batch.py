"""Independently dispatch the chosen entry cases and verify the parallel transfers."""
from fit import *
def main():
    tic=time.monotonic();payload=[]
    if len(CASES)==5:CASES.append(('acidic',1.))
    fits=pd.concat([pd.read_csv(OUT/'frozen-final.csv'),pd.read_csv(OUT/'acidic-frozen.csv')]);archive=np.load(OUT/'final-spectra.npz');trace=np.load(OUT/'final-gas.npz')
    for row in fits.sort_values('train_mse').drop_duplicates('entry').itertuples():
        p=np.array([getattr(row,key) for key in NAMES]);k=int(row.k);entry=int(row.entry);key=f'test_288_{k}_{entry}_1_80'
        s,q=run(288,k,'test',entry,True,p)
        s=s[0,0].astype('f4');q=q[0,0]
        np.testing.assert_allclose(s,archive[key],rtol=2e-7,atol=1e-8)
        np.testing.assert_allclose(q,trace[key],rtol=2e-12,atol=1e-9)
        payload.append(dict(key=key,max_density_delta=float(abs(s-archive[key]).max()),max_acid_delta=float(abs(q[:,0]-trace[key][:,0]).max())))
    (OUT/'batch-validation.json').write_text(json.dumps(dict(seconds=time.monotonic()-tic,results=payload),indent=2))
if __name__=='__main__':main()
