"""Estimate changes in the source effect under paired changes of forcing."""
import pandas as pd
from scipy.stats import t
from campaign import DUST, BASE_SEED
from summarize import lookup


def write(d,out):
    rows=[]
    choices=[('mix2','reference'),('mix8','reference'),
             ('synthetic_roll200_1800','mix2'),('synthetic_roll400_1800','mix2'),('synthetic_roll200_600','mix2'),
             ('isdac_roll200_1800','isdac'),('mpace_roll200_1800','mpace'),
             ('isdac_we4p125','isdac'),('mpace_we8p2','mpace'),('qplus050','qbase'),
             *[('mpace_flux'+str(v),'mpace') for v in [5,10,25,50,100]]]
    for new,old in choices:
        for abundance in ['equal','air_volume']:
            for source in [DUST,'Marine','RuralContinental']:
                for control in ['aged_'+DUST,'closed']:
                    pairs=[]
                    for sky in [new,old]:
                        a=lookup(d,sky=sky,donor=source,abundance=abundance).set_index('seed')
                        b=lookup(d,sky=sky,donor=control,abundance=abundance).set_index('seed')
                        pairs.append(a[['primary','unique','suspended','LWP']]-b[['primary','unique','suspended','LWP']])
                    joined=(pairs[0]-pairs[1]).dropna()
                    spaced=joined.loc[joined.index>=BASE_SEED+10000019]
                    if len(spaced):joined=spaced
                    if not len(joined):continue
                    n=len(joined)
                    for metric,x in joined.items():
                        mean=x.mean()
                        half=t.ppf(.975,n-1)*x.sem() if n>1 else float('nan')
                        rows.append(dict(new=new,old=old,source=source,control=control,abundance=abundance,metric=metric,
                                         n=n,new_effect=pairs[0].loc[x.index,metric].mean(),
                                         old_effect=pairs[1].loc[x.index,metric].mean(),change=mean,lo=mean-half,hi=mean+half))
    pd.DataFrame(rows).to_csv(out/'interactions.csv',index=False)
