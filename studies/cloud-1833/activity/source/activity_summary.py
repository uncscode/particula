# %% Paired stage uncertainty and study figures from preserved numerical spectra.
from pathlib import Path
from physics import *
from activity import OUT
def main():
    files=np.load(OUT/"spectra.npz");ix=segments['test'];valid=(stage[ix]<18);valid[0]=False
    ref=files["test_288_original_1.0_attractive"].astype(float);o=y[ix]
    weights=np.diff(e)[4:];wt=weights/weights.sum()
    def err(z):return abs(np.log10(1+z[:,4:])-np.log10(1+o[:,4:]))@wt
    rng=np.random.default_rng(183311);draw=rng.integers(0,9,(10000,9))
    num=np.bincount(stage[ix][valid]-9,minlength=9);rows=[]
    for key in files:
        if not key.startswith("test"):continue
        delta=err(files[key].astype(float))-err(ref);part=np.bincount(stage[ix][valid]-9,weights=delta[valid],minlength=9)
        boot=10**(part[draw].sum(1)/num[draw].sum(1));rows.append((key,10**delta[valid].mean(),*np.quantile(boot,[.025,.975]),np.sum(part<0)))
    pd.DataFrame(rows,columns=["case","error_factor_ratio_to_original","stage_p025","stage_p975","stages_improved"]).to_csv(OUT/"comparison.csv",index=False)
    from importlib.util import module_from_spec, spec_from_file_location
    script=Path(__file__).resolve().parents[1]/"plot.py"
    spec=spec_from_file_location("water_plot",script)
    plotting=module_from_spec(spec)
    spec.loader.exec_module(plotting)
    plotting.plot(OUT, OUT/"spectra.npz", OUT/"activity-results.png")
if __name__=='__main__':main()
