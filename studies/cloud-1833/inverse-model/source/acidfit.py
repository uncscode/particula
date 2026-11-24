from fit import *
from forward import CASES
def main():
    t0=time.monotonic();assert (PACKET/"activity/acid-freeze.json").exists()
    if len(CASES)==5:CASES.append(("acidic",1.))
    rows=[];fine=[]
    for entry in (0,1):
        for j,x in enumerate([CENTER.copy(),np.array([.18,np.log(.9),np.log(1.14),np.log(1.2),np.log(3.),np.log(.4)])]):
            p,l,c,s,o=exactfit(5,entry,x,64,18)
            rows.append(dict(k=5,entry=entry,start=j,train_mse=l,calls=c,stationarity=o,**dict(zip(NAMES,p))))
        best=min(rows[-2:],key=lambda r:r['train_mse']);init=np.array([best[n] for n in NAMES])
        z,loss,c,sv,opt=exactfit(5,entry,init,192,16)
        fine.append(dict(k=5,entry=entry,start=best['start'],train_mse=loss,calls=c,stationarity=opt,min_singular=sv[-1],**dict(zip(NAMES,z))))
        print('acidic',entry,loss,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'acidic-fits.csv',index=False);pd.DataFrame(fine).to_csv(OUT/'acidic-frozen.csv',index=False)
    (OUT/'acidic-runtime.json').write_text(json.dumps(dict(seconds=time.monotonic()-t0,calls=sum(r['calls'] for r in rows+fine))))
if __name__=="__main__":main()
