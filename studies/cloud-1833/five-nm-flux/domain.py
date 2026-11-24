"""Five-nanometer domain from the parent sulfate-conserving geometry."""
from physics import (lru_cache, np, e, solid_mass, salt, amu, capture, par, T, P, overlap, reported, study, d)
@lru_cache(maxsize=None)
def geometry(n,law="original",retain=1.,legacy=False):
    edges=np.linspace(np.log10(5.),e[-1] if legacy else np.log10(96.),n+1)
    dx=edges[1]-edges[0]; D=10**edges;mid=10**((edges[1:]+edges[:-1])/2)
    salt_mw=98.079+1.5*17.03052 if law=="acidic" else salt
    der=(solid_mass(D*(1+2e-5),law)-solid_mass(D*(1-2e-5),law))/(4e-5*D*salt_mw*amu)
    cap=capture(D); vel=cap/der/(D*np.log(10))
    if legacy:
        dx,edges,base,wall,K,pairs,over,mi,mo=study.grid(n,d,e)
        return dict(edges=edges,D=mid,dx=dx,vel=base[[2,1]],wall=wall,K=K,pairs=pairs,over=over,
                    mi=mi,mo=mo,cap=capture(mid),mapped=edges)
    wall=.002*np.sqrt(par.particles.get_diffusion_coefficient_via_system_state(mid*.5e-9,T,P)/(8e-6*(T/293.15)**1.75))
    m=solid_mass(mid,law);physical_mass=(1770 if law=="dry" else 1350)*np.pi/6*(mid*1e-9)**3
    K=par.dynamics.get_brownian_kernel_via_system_state(mid*.5e-9,physical_mass,T,P)*1e6
    i,j=np.triu_indices(n); total=m[i]+m[j];hi=np.searchsorted(m,total);ok=hi<n
    i,j,total,hi=(a[ok] for a in (i,j,total,hi)); f=(total-m[hi-1])/(m[hi]-m[hi-1])
    k=K[i,j]*np.where(i==j,.5,1);dest=np.r_[hi,hi-1];order=np.argsort(dest)
    pairs=[np.r_[0,np.bincount(dest,minlength=n).cumsum()].astype('i4'),np.tile(i,2)[order].astype('i4'),
           np.tile(j,2)[order].astype('i4'),np.r_[k*f,k*(1-f)][order]]
    over=np.searchsorted(m,m[-1]-m,side="right").astype('i4')
    # Audit coagulation on the conserved sulfate mass coordinate.
    probe=np.exp(-((np.log(mid/12))/.55)**2)
    pair=K*probe[:,None]*probe; excess=m[:,None]+m>m[-1]
    gain=np.bincount(dest[order],weights=pairs[3]*probe[pairs[1]]*probe[pairs[2]],minlength=n)
    assert abs((gain-pair.sum(1))@m+.5*np.sum(pair*excess*(m[:,None]+m)))<1e-10*(pair.sum(1)@m)
    mapped=np.log10(reported(D,law,retain))
    mi=overlap(mapped[:-1],mapped[1:],e)
    mo=overlap(e[:-1],e[1:],mapped)/np.diff(mapped)
    assert np.all(mo.sum(0)<=1+1e-11) and np.all(np.diff(mapped)>0)
    return dict(edges=edges,D=mid,dx=dx,vel=vel,wall=wall,K=K,pairs=pairs,over=over,
                mi=mi,mo=mo,cap=capture(mid),mapped=mapped)
