# %% Secondary sensitivity to a prescribed acid-rich analytical inventory.
import json,time
from activity import *
def main():
    start=time.monotonic();rows=[];details={};budget=[]
    for n in [192,288]:
        for seg,ix in segments.items():
            spec,cs,err=replay(n,"acidic",1.,ix);budget.append(err)
            for c,name in enumerate(["hard","attractive"]):
                ok=~missing[ix];key=f"{seg}_{n}_acidic_1.0_{name}"
                for m in metrics(spec[c,ok],ix[ok]):
                    rows.append(dict(key=key,section=n,hydration="acidic",retained=1.,capture=name,segment=seg,
                                resolved_sink_s=cs[c,ok].mean(),**m))
                if n==288:details[key]=spec[c].astype("f4")
    pd.DataFrame(rows).to_csv(OUT/"acidic-scores.csv",index=False)
    np.savez_compressed(OUT/"acidic-spectra.npz",**details)
    sizes=np.array([2.9,4.,9.17,10.,20.,64.68])
    pd.DataFrame(dict(diameter_nm=sizes,water_mass_fraction=fraction(sizes,"acidic"))).to_csv(OUT/"acidic-water.csv",index=False)
    (OUT/"acid-freeze.json").write_text(json.dumps(dict(scenario="acidic liquid, 1.5 NH3 per sulfate, measured wet, independent sensitivity",cuda_boxes=8,max_balance_error=max(budget),seconds=time.monotonic()-start,decision="Carry this specified acidic sensitivity into both entry and chamber comparisons. Composition is an assumption."),indent=2))
if __name__=='__main__':main()
