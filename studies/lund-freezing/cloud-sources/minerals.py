"""Persistent mineral active sites on incoming grains, with rare-event stratification."""
import numpy as np
import warp as wp
import physics as gpu

COEFFICIENTS = {
    'feldspar': (-3.25, -.793, -.0691, -.00417, -.000105, -.000000908),
    'quartz': (-1.709, .07, .0175, .000266),
}


def site_density(species, temperature):
    t = np.asarray(temperature)
    warm_limit = -10.5 if species == 'quartz' else -3.5
    if np.min(t) < -37.5 or np.max(t) > warm_limit:
        raise ValueError('The cloud profile exceeds the Harrison et al. temperature range')
    return 10.**np.polynomial.polynomial.polyval(t, COEFFICIENTS[species])


@wp.kernel
def mineral_freeze(state:wp.array(dtype=gpu.State), draws:gpu.F, probability:gpu.F,
                   clock:gpu.F, dt:wp.float64):
    i=wp.tid()
    a=state[i]
    if a.fate!=1 or a.phase!=1 or draws[i]>probability[a.cell]:
        return
    a.phase=2
    a.freezes+=1
    a.frozen_mass+=a.mass
    if a.dries>0:
        a.recycled_ice+=1
    if a.first<wp.float64(0.):
        a.first=clock[0]+dt
        a.firstx=a.pos[0]
        a.firstz=a.pos[1]
    state[i]=a


class Mineral:
    def __init__(self,species,diameter,grain_loading,log_shift,seed):
        self.direct=species.startswith('direct_')
        self.sham=species.startswith('sham_')
        self.species=species.removeprefix('direct_').removeprefix('sham_')
        self.diameter=diameter             # micrometres
        self.grain_loading=grain_loading   # incoming grains per L
        self.log_shift=log_shift           # log10 site-density shift
        self.seed=seed

    def initialize(self,original,saturation,device):
        inlet=(original['carrier']==1)&(original['birth']>=0)
        original['dry_radius'][inlet]=self.diameter*0.5e-6
        original['weight'][inlet]*=self.grain_loading
        original['rate'][inlet]=0.
        nominal=original['weight'][inlet].sum()
        # Use equal prescribed accessible area per grain. The primary source
        # reports BET area; spherical geometric area is an explicit assumption.
        area=np.pi*(self.diameter*1e-4)**2    # cm2
        if self.species=='inert':
            probability=np.zeros(len(saturation))
        else:
            ns=site_density(self.species,saturation[:,4])*10.**self.log_shift
            probability=-np.expm1(-area*ns)
        maximum=probability.max()
        self.maximum=float(maximum)
        self.grains_per_litre=float(nominal/(1600*600*1000))
        if self.direct:
            a=original
            draws=np.full(len(a),2.)
            draws[inlet]=np.random.default_rng(self.seed+310087).random(inlet.sum())
        elif maximum>0.:
            active=original[inlet].copy()
            active['weight']*=maximum
            original['weight'][inlet]*=1.-maximum
            a=np.empty(len(original)+len(active),dtype=original.dtype)
            a[:len(original)]=original
            a[len(original):]=active
            rng=np.random.default_rng(self.seed+310087)
            draws=np.r_[np.full(len(original),2.),maximum*rng.random(len(active))]
            assert np.isclose(a['weight'][(a['carrier']==1)&(a['birth']>=0)].sum(),nominal)
        else:
            a=original
            draws=np.full(len(a),2.)
        if self.sham:
            draws[:]=2.  # Preserve both strata and paths with mineral freezing switched off.
        self.device=device
        self.draws=wp.array(draws,dtype=wp.float64,device=device)
        self.prob=wp.array(probability,dtype=wp.float64,device=device)
        wp.load_module(module=mineral_freeze.module,device=device)
        return a

    def freeze(self,state,clock,dt):
        wp.launch(mineral_freeze,len(state),inputs=[state,self.draws,self.prob,clock,dt],device=self.device)
