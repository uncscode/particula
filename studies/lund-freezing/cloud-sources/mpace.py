"""Ice-free saturation adjustment for Klein et al. (2009), equations 1–3."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from isdac import RD, RV, CP, LV, GRAVITY, saturation


def layer(p):
    hpa=p/100.
    if hpa>850.:
        theta,qt=269.2,.00195
    else:
        theta,qt=275.33+.0791*(815.-hpa),.001*(.291+.00204*(hpa-590.))
    dry=theta*(p/100000.)**(RD/CP)
    def liquid(t):
        es=saturation(t)
        return max(0.,qt-RD/RV*es/(p-es))
    t=brentq(lambda t:t*np.exp(-LV*liquid(t)/(CP*263.))-dry,dry,dry+8) if liquid(dry)>0 else dry
    ql=liquid(t)
    qv=qt-ql
    rho=p/((RD+RV*qv)*t)
    return t,p,rho*qv,rho*ql,rho*(1+qt),theta,qt


@lru_cache(None)
def sounding():
    rows=[];p=101000.
    for z in range(1401):
        r=layer(p)
        rows.append(r)
        p-=GRAVITY*layer(p-.5*GRAVITY*r[4])[4]
    return np.asarray(rows)


def column(z):
    table=sounding();alt=np.asarray(z)+732.
    t,p,v,l=[np.interp(alt,np.arange(len(table)),table[:,j]) for j in range(4)]
    return t-273.15,p,v,l


def inlet():
    return float(sounding()[1350,2])
