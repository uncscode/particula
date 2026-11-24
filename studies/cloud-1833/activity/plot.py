"""Water uptake, measured count and size-resolved residuals from three retained runs."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd


def plot(directory, spectra, output):
    directory=Path(directory)
    study=Path(__file__).resolve().parents[1]
    with np.load(study/"spectra.npz") as data:
        t,d,e,y,stage=(data[k] for k in ["time","diameter","edges","observed","stage"])
    missing=pd.read_csv(study/"lamp_budget.csv").small_missing.to_numpy()
    ix=np.flatnonzero((stage>=9)&(stage<=18))
    o=y[ix]
    with np.load(spectra) as data:
        files={key:data[key] for key in data if key.startswith("test_288_aiomfac_") and key.endswith("_attractive")}
    paper_time=t[ix]+6.5;tail=np.maximum(0,e[1:]-np.maximum(e[:-1],np.log10(9.7)))
    fig,ax=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
    curve=pd.read_csv(directory/'activity.csv');a=ax[0,0]
    a.plot(curve.diameter_nm,100*curve.water_mass_fraction,label='AIOMFAC liquid',color='#08776B')
    a.plot(curve.diameter_nm,100*curve.original_water_mass_fraction,label='Original hydration',color='#535e87',ls='--')
    a.set(xscale='log',xlabel='Wet diameter (nm)',ylabel='Water mass (%)',title='Water gained as sulfate particles grow');a.legend(frameon=False)
    a=ax[0,1];a.plot(paper_time,o@tail,color='black',label='Measured',lw=1.5)
    for r,c,ls in [(1.,'#08776B','-'),(.5,'#e08520','--'),(0.,'#87579a',':')]:
        key=f'test_288_aiomfac_{r}_attractive'
        a.plot(paper_time,files[key]@tail,color=c,ls=ls,lw=1.5,label=f'{r:.0%} water retained in sizing')
    a.set(xlabel='Time from run start (hours)',ylabel='Particles ≥9.7 nm (cm⁻³)',title='Size-selected concentration');a.legend(frameon=False,fontsize=8)
    for a,r in zip(ax[1],[1.,.5]):
        pred=files[f'test_288_aiomfac_{r}_attractive']
        residual=(pred-o).copy()
        residual[np.ix_(missing[ix], np.arange(7))]=np.nan
        img=a.pcolormesh(paper_time,d[4:],residual[:,4:].T,cmap='RdBu_r',shading='nearest',norm=SymLogNorm(100,vmin=-40000,vmax=40000))
        a.set(yscale='log',xlabel='Time from run start (hours)',ylabel='Measured diameter (nm)',title=f'Predicted − measured density, {r:.0%} retention')
    fig.colorbar(img,ax=ax[1],label='Particles cm⁻³ per decade');fig.savefig(output,dpi=155);plt.close(fig)


if __name__ == "__main__":
    directory=Path(__file__).resolve().parent
    plot(directory, directory/"plot-spectra.npz", sys.argv[1])
