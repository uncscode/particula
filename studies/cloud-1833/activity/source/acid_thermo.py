"""Charge- and sulfur-balanced liquid at an imposed 1.5 ammonia per sulfate."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import PchipInterpolator
import particula_aiomfac as ai
SALT=98.079+1.5*17.03052
MODEL=ai.AIOMFACModel.prepare(ai.MixtureSpec({"water":{16:1},"bisulfate":{204:1,248:1},"sulfate":{204:2,261:1}}))
PART=(98.079+17.03052)/(2*SALT)
def evaluate(f):
    m=np.stack([f,(1-f)*PART,(1-f)*(1-PART)],axis=-1)
    r=MODEL.evaluate(m,297.8,diagnostics=True)
    assert np.all(r.solver_status==1)
    assert np.max(abs(r.diagnostics["equilibrium_residuals"]))<1e-7
    return r.neutral_activities[...,0]
@lru_cache(None)
def seed():
    f=np.linspace(.07,.45,70);aw=evaluate(f)
    return PchipInterpolator(aw,f),PchipInterpolator(f,aw).derivative()
_cache={}
def fraction(D):
    kelvin=4*.079*.01801528/(1000*8.314462618*297.8)*1e9
    new=np.unique([v for v in D if v not in _cache]);init,der=seed()
    if len(new):
        target=.58*np.exp(-kelvin/new);f=init(target)
        for _ in range(4):
            residual=evaluate(f)-target
            if abs(residual).max()<1e-10:break
            f-=residual/der(f)
        assert abs(residual).max()<1e-10
        _cache.update(zip(new,f))
    return np.array([_cache[v] for v in D])
