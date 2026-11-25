"""Thermodynamic initialization from Ovchinnikov et al. 2014, Appendix A1/A2."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq

RD, RV, CP, LV, GRAVITY = 287.05, 461.5, 1004., 2.5e6, 9.81


def saturation(t):
    return np.exp(54.842763-6763.22/t-4.210*np.log(t)+.000367*t+
                  np.tanh(.0415*(t-218.8))*(53.878-1331.22/t-9.44523*np.log(t)+.014025*t))


def layer(z, p):
    if z < 400:
        theta_l, qt = 265+.004*(z-400), .0015-.00000075*(z-400)
    elif z < 825:
        theta_l, qt = 265., .0015
    else:
        theta_l, qt = 266.+(z-825)**.3, .0012
    exner=(p/100000.)**(RD/CP)
    def liquid(t):
        es=saturation(t)
        return max(0.,qt-(RD/RV)*es/(p-es))
    dry=theta_l*exner
    t=brentq(lambda t:t*np.exp(-LV*liquid(t)/(CP*t))-dry,dry,dry+LV*qt/CP+0.05) if liquid(dry)>0 else dry
    ql=liquid(t)
    qv=qt-ql
    rdensity=p/((RD+RV*qv)*t)
    return t,p,rdensity*qv,rdensity*ql,rdensity*(1+qt),theta_l,qt


@lru_cache(None)
def sounding():
    # Hydrostatic integration from the published surface pressure (1020 hPa).
    rows=[]
    p=102000.
    for z in range(901):
        values=layer(z,p)
        rows.append(values)
        midpoint=layer(z+.5,p-.5*GRAVITY*values[4])
        p-=GRAVITY*midpoint[4]
    return np.asarray(rows)


def column(z):
    table=sounding()
    alt=np.asarray(z)+225.
    t,p,v,l=[np.interp(alt,np.arange(len(table)),table[:,j]) for j in range(4)]
    return t-273.15,p,v,l


def inlet():
    return float(sounding()[850,2])
