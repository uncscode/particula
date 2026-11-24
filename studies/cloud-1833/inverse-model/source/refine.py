"""Freeze refined fits before scoring the transfer stages."""
from fit import *
def main():
    start=time.monotonic();df=pd.read_csv(OUT/"fits.csv")
    records=[]
    for (k,entry),group in df.groupby(["k","entry"]):
        group=group.sort_values("train_mse");a=group.iloc[0];starts=[a]
        b=group.iloc[1];nf=6 if entry else 4
        if b.train_mse-a.train_mse<=.025 and max(abs((a[NAMES[:nf]]-b[NAMES[:nf]]).to_numpy()))>.2:starts.append(b)
        for row in starts:
            z,loss,calls,sv,opt=exactfit(k,entry,row[NAMES].to_numpy(float),192,16)
            records.append(dict(k=k,entry=entry,start=int(row.start),train_mse=loss,calls=calls,stationarity=opt,min_singular=sv[-1],**dict(zip(NAMES,z))))
        pd.DataFrame(records).to_csv(OUT/"refined-training.csv",index=False)
        print('refined',k,entry,records[-1]['train_mse'],round(time.monotonic()-start,1),flush=True)
    df=pd.DataFrame(records);df.loc[df.groupby(["k","entry"]).train_mse.idxmin()].to_csv(OUT/"frozen-final.csv",index=False)
    (OUT/"refine-runtime.json").write_text(json.dumps(dict(seconds=time.monotonic()-start,calls=int(df.calls.sum()))))
if __name__=='__main__':main()
