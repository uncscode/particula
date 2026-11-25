"""CUDA superparticle trajectories, phase transitions and local water exchange."""
import warp as wp

D=wp.float64
F=wp.array(dtype=wp.float64)
M=wp.array2d(dtype=wp.float64)
I=wp.array(dtype=wp.int32)
PI=wp.constant(3.141592653589793)
LENGTH=wp.constant(1600.0)
DEPTH=wp.constant(600.0)

@wp.struct
class State:
    pos:wp.vec2d
    mass:D                # water mass per represented particle, kg
    clock:D               # integrated stochastic hazard
    threshold:D           # exponential draw until next freeze
    first:D               # time of first ice, s
    firstx:D
    firstz:D
    rate:D                # per hour at -14 C
    birth:D               # negative for residents, absolute seconds for arrivals
    weight:D              # physical multiplicity per tracked record
    fate:int              # 0 pending, 1 airborne, 2 top export, 3 fallout
    carrier:int           # 0 background CCN, 1 Lund carrier
    phase:int             # 0 aerosol/haze, 1 liquid >1 um, 2 ice
    wets:int              # entries into liquid phase
    dries:int             # liquid to haze transitions
    freezes:int
    recycled_ice:int      # freezes after a previous wet/dry cycle
    sublimations:int
    frozen_mass:D         # cumulative liquid mass initially frozen, kg
    evap:D                # cumulative liquid evaporation, kg
    cond:D
    dep:D                 # vapor to existing ice
    sub:D
    lastwet:D             # wet exposure accumulated since last drying
    wettime:D             # cumulative immersed time
    cell:int
    change:D
    dry_radius:D

@wp.func
def velocity(p:wp.vec2d, W:D, E:D):
    k=D(2.0)*D(PI)/D(LENGTH)
    eta=p[1]/D(DEPTH)
    phase=k*p[0]
    u=-wp.sin(phase)*(W*D(PI)*wp.cos(D(PI)*eta)+D(3.)*E*eta*eta)/(k*D(DEPTH))
    w=wp.cos(phase)*(W*wp.sin(D(PI)*eta)+E*eta*eta*eta)
    return wp.vec2d(u,w)

@wp.func
def radius(m:D,rho:D):
    return wp.pow(wp.max(m,D(0.))*D(3.)/(D(4.)*D(PI)*rho),D(1.)/D(3.))

@wp.func
def terminal(r:D,rho:D):
    stokes=D(2.)*rho*D(9.81)*r*r/(D(9.)*D(1.65e-5))
    v=stokes
    for j in range(5):
        re=D(2.)*r*v*D(1.1)/D(1.65e-5)
        v=stokes/(D(1.)+D(.15)*wp.pow(re,D(.687)))
    return v

@wp.func
def bin_id(p:wp.vec2d,n:int):
    ix=wp.clamp(int(p[0]*D(n)/D(LENGTH)),0,n-1)
    iz=wp.clamp(int(p[1]*D(n)/D(DEPTH)),0,n-1)
    return ix*n+iz

@wp.kernel
def move(s:wp.array(dtype=State),clock:F,dt:D,n:int,W:D,E:D,fall:D):
    i=wp.tid(); a=s[i]; t=clock[0]
    if a.fate>=2 or a.birth>=t+dt:
        return
    h=dt
    if a.fate==0:
        a.fate=1
        h=t+dt-wp.max(a.birth,t)
    density=D(1000.)
    if a.phase==2:
        density=D(917.)
    slip=fall*terminal(radius(a.mass,density),density)
    drift=wp.vec2d(D(0.),slip)
    v=velocity(a.pos,W,E)-drift
    new=a.pos+h*(velocity(a.pos+D(.5)*h*v,W,E)-drift)
    x=new[0]-D(LENGTH)*wp.floor(new[0]/D(LENGTH))
    a.pos=wp.vec2d(x,new[1])
    if new[1]>=D(DEPTH):
        a.fate=2
    elif new[1]<D(0.):
        a.fate=3
    else:
        a.cell=bin_id(a.pos,n)
    s[i]=a

@wp.kernel
def local_air(v:M,water:F,sat:M,clock:F,dt:D,n:int,volume:D,restore:F):
    j=wp.tid()
    z=(D(j%n)+D(.5))*D(DEPTH)/D(n)
    # Specified lower-boundary humidity support, with its own mass ledger.
    if z<D(120.):
        dv=(D(.985)*sat[j,0]*volume-v[j,0])*(D(1.)-wp.exp(-dt/D(900.)))
        v[j,0]+=dv
        restore[j]+=dv
    water[j]=D(0.)

@wp.kernel
def propose(s:wp.array(dtype=State),v:M,sat:M,volume:D,dt:D,clock:F,ice:int,total:F):
    i=wp.tid(); a=s[i]
    if a.fate!=1 or (ice==1 and a.phase!=2) or (ice==0 and a.phase==2):
        return
    h=wp.min(dt,clock[0]+dt-a.birth) if a.birth>=D(0.) else dt
    j=a.cell; ratio=v[j,0]/(volume*sat[j,ice])
    rho=D(1000.); G=sat[j,2]
    if ice==1:
        rho=D(917.);G=sat[j,3]
    rd=a.dry_radius
    r=wp.pow(a.mass*D(3.)/(D(4.)*D(PI)*rho)+rd*rd*rd,D(1.)/D(3.))
    # Inactive dry cores require a prescribed 0.05% supersaturation for reactivation.
    if a.phase==0 and a.mass==D(0.) and ratio<D(1.0005):
        a.change=D(0.)
    else:
        r2=wp.max(rd*rd,r*r+D(2.)*G*(ratio-D(1.))*h)
        m=wp.max(D(0.),D(4.)*D(PI)*rho/D(3.)*(r2*wp.sqrt(r2)-rd*rd*rd))
        if r2<=rd*rd*D(1.00000001):
            m=D(0.)
        a.change=m-a.mass
    wp.atomic_add(total,j,a.weight*a.change)
    s[i]=a

@wp.kernel
def limit(v:M,sat:M,total:F,scale:F,volume:D,ice:int):
    j=wp.tid()
    delta=total[j]
    scale[j]=D(1.)
    if wp.abs(delta)>D(0.):
        scale[j]=wp.min(D(1.),wp.abs(v[j,0]-sat[j,ice]*volume)/wp.abs(delta))
    v[j,0]-=delta*scale[j]
    total[j]=D(0.)

@wp.kernel
def phase_step(s:wp.array(dtype=State),scale:F,sat:M,clock:F,dt:D,ice:int,beta:D,
               shape:D,reset:int,seed:int):
    i=wp.tid();a=s[i]
    if a.fate!=1 or (ice==1 and a.phase!=2) or (ice==0 and a.phase==2):
        return
    dm=a.change*scale[a.cell]
    a.mass=wp.max(D(0.),a.mass+dm)
    if ice==1:
        a.dep+=wp.max(D(0.),dm);a.sub+=wp.max(D(0.),-dm)
        if a.mass<=D(1.e-28):
            a.phase=0;a.sublimations+=1
            a.lastwet=D(0.)
    else:
        a.cond+=wp.max(D(0.),dm);a.evap+=wp.max(D(0.),-dm)
        if radius(a.mass,D(1000.))>=D(.000001):
            if a.phase==0:
                a.wets+=1
            a.phase=1
            h=wp.min(dt,clock[0]+dt-a.birth) if a.birth>=D(0.) else dt
            # Integrate rate over effective wet time at a prescribed thermal slope.
            thermal=wp.exp(-beta*(sat[a.cell,4]+D(14.)))
            t0=a.lastwet/D(3600.)
            t1=t0+h*thermal/D(3600.)
            a.clock+=a.rate*(wp.pow(t1,shape)-wp.pow(t0,shape))
            a.lastwet=t1*D(3600.);a.wettime+=h
            if a.clock>=a.threshold:
                a.phase=2;a.freezes+=1;a.frozen_mass+=a.mass
                if a.dries>0:
                    a.recycled_ice+=1
                if a.first<D(0.):
                    a.first=clock[0]+dt;a.firstx=a.pos[0];a.firstz=a.pos[1]
                rng=wp.rand_init(seed+i,a.freezes)
                a.threshold=-wp.log(wp.max(D(1.e-12),D(wp.randf(rng))))
                a.clock=D(0.)
        elif a.phase==1:
            a.phase=0;a.dries+=1
            if reset==1:
                a.lastwet=D(0.)
    s[i]=a

@wp.kernel
def tick(clock:F,dt:D):
    clock[0]+=dt

@wp.kernel
def sample_ids(s:wp.array(dtype=State),ids:I,out:wp.array(dtype=State)):
    j=wp.tid()
    out[j]=s[ids[j]]

@wp.kernel
def disperse(s:wp.array(dtype=State),clock:F,dt:D,n:int,K:D,seed:int):
    i=wp.tid();a=s[i]
    if a.fate!=1:
        return
    h=wp.min(dt,clock[0]+dt-a.birth) if a.birth>=D(0.) else dt
    rng=wp.rand_init(seed+i,int(clock[0]/dt)+9137)
    sigma=wp.sqrt(D(2.)*K*h)
    x=a.pos[0]+sigma*D(wp.randn(rng))
    z=a.pos[1]+sigma*D(wp.randn(rng))
    # Only advective flow and particle fall speed cross top/bottom. Reflect
    # the unresolved mixing increment, matching the zero exterior grid diffusion.
    z=wp.abs(z)
    if z>D(DEPTH):
        z=D(2.)*D(DEPTH)-z
    a.pos=wp.vec2d(x-D(LENGTH)*wp.floor(x/D(LENGTH)),z)
    a.cell=bin_id(a.pos,n)
    s[i]=a
