"""Bootstrap entire excimer stages for matched capture and hydration comparisons."""
from activity import *
def main():
    results=pd.concat([pd.read_csv(OUT/'scores.csv'),pd.read_csv(OUT/'acidic-scores.csv')])
    table=results.query("segment=='test' and stage>=9 and stage<=17").pivot(index='key',columns='stage',values='log_error')
    idx=segments['test'];st=stage[idx][1:]
    w=np.array([(st==j).sum() for j in range(9,18)])
    draws=np.random.default_rng(183311).integers(0,9,(10000,9))
    rows=[]
    for key,x in table.iterrows():
        seg,n,law,retained,capture=key.split('_')
        for contrast,other in [('same_capture_original',f'test_{n}_original_1.0_{capture}'),('same_water_hard',f'test_{n}_{law}_{retained}_hard')]:
            delta=x.to_numpy()-table.loc[other].to_numpy();boot=10**((delta*w)[draws].sum(1)/w[draws].sum(1))
            rows.append(dict(case=key,contrast=contrast,reference=other,ratio=10**np.average(delta,weights=w),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975),stages_improved=sum(delta<0)))
    pd.DataFrame(rows).to_csv(OUT/'all-stage-contrasts.csv',index=False)
if __name__=='__main__':main()
