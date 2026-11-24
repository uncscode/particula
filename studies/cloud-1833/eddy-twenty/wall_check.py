"""Rate profile and analytic diffusion-limit check against the earlier CLOUD model."""
from pathlib import Path
import numpy as np
import pandas as pd
from treatments import coefficient, DIAMETER, HEIGHT, VOLUME
from physics import par,T,P

nm=np.array([5.,7.7,9.7,10.,15.,25.,40.,63.5,96.])
diff=par.particles.get_diffusion_coefficient_via_system_state(nm*.5e-9,T,P)
base=.002*np.sqrt(diff/(8e-6*(T/293.15)**1.75))
old=.35*base
factor=4/np.pi*(2/DIAMETER+1/HEIGHT)
equivalent=(old/np.sqrt(diff)/factor)**2
rows=[]
for rate in [equivalent[0],.045,.05,.0525,.15]:
    k=coefficient(nm,{'eddy':rate},base)
    diff_limit=factor*np.sqrt(rate*diff)
    assert np.all(k>=diff_limit*(1-1e-12))
    rows.append(pd.DataFrame(dict(diameter_nm=nm,eddy_s_inv=rate,native_loss_s_inv=k,
        diffusion_limit_s_inv=diff_limit,earlier_loss_s_inv=old,
        native_wall_lifetime_minutes=1/(60*k))))
pd.concat(rows).to_csv(Path(__file__).with_name('wall-rates.csv'),index=False)
print(f'Diffusion-equivalent eddy rate for the previous multiplier {equivalent[0]:.8f} per second')
