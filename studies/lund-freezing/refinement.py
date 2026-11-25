# %%
"""Refine mixture fitting, challenge pooling, and compare late forecast errors."""
import argparse
from pathlib import Path
import pandas as pd
from study import read_inputs,geometry,train,predictions,score
from challenges import summarize


def main(pkt,out):
    d=geometry(read_inputs(pkt,False)[0]);rows=[]
    for name,g in d.groupby('sample'):
        for model,b,k in [('persistent',1,0),('aging',.6,0),('redrawn',1,.6)]:
            for tag,n,it,shrink in [('primary',21,180,16),('grid_and_steps',41,720,16),
                                    ('more_steps',41,1600,16),('weak_pool',41,720,4),
                                    ('strong_pool',41,720,64)]:
                fit=train(g,nrate=n,power=b,gamma=k,steps=it,strength=shrink)
                p=predictions(g,fit,power=b,gamma=k)
                for row in score(p,tag):
                    if row['subset']=='all' and row['model']=='persistent':
                        row.update(law=model);rows.append(row)
    s=pd.DataFrame(rows);s.to_csv(out/'estimator_refinement.csv',index=False)
    fixed=s[s.variant=='primary'].pivot(index=['sample','plate'],columns='law',values='abs_error')
    df=pd.DataFrame({'aging_over_redrawn':fixed.redrawn-fixed.aging}).reset_index()
    print(summarize(df,['aging_over_redrawn'],'all'))
    print(s.groupby(['law','variant'])[['predicted','id_gain_early','abs_error']].mean())
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('input',type=Path);a.add_argument('output',type=Path)
    args=a.parse_args();main(args.input,args.output)
