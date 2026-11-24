"""Check predicted entry against refinement of the sub-2.9 nm transport cells."""
from fit import *
from forward import NC
def main():
    start=time.monotonic()
    rows=[]
    fits=pd.concat([pd.read_csv(OUT/'frozen-final.csv'),pd.read_csv(OUT/'acidic-frozen.csv')])
    r=fits.query('entry==1').sort_values('train_mse').iloc[0]
    if len(CASES)==5:CASES.append(('acidic',1.))
    k=int(r.k);p=r[NAMES].to_numpy(float)
    for n in [192,288]:
        for two in [0,1]:
            s,q=run(n,k,'test',True,two,p);ix=segments['test'];ok=~missing[ix]
            for m in metrics(s[0,0,ok],ix[ok]):
                mask=(stage[ix]==m['stage']) if m['stage'] else (stage[ix]<18)
                mask[0]=False;mask&=~missing[ix]
                rows.append(dict(n=n,k=k,cluster_cells=NC,two=two,entry_cm3s=q[0,0,mask,1].mean(),
                    mean_2p9_density=s[0,0,mask,3].mean(),**m))
    pd.DataFrame(rows).to_csv(OUT/f'cluster-grid-{NC}.csv',index=False)
    (OUT/f'cluster-grid-{NC}-runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,boxes=4)))
if __name__=='__main__':main()
