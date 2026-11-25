"""Conservative, oscillating circulation and explicit lower water supply on CUDA."""
import numpy as np
import warp as wp
import physics as micro

D=wp.float64
M=wp.array2d(dtype=wp.float64)
PERIODIC_LENGTH=wp.constant(1600.)
HEIGHT=wp.constant(600.)

@wp.func
def offset(clock:D,amplitude:D,period:D):
    return amplitude*wp.sin(D(2.)*D(micro.PI)*clock/period)

@wp.kernel
def surfaces(base:M,sine:M,cosine:M,mixing:M,inward:M,outward:M,clock:micro.F,
             dt:D,amplitude:D,period:D):
    j,f=wp.tid()
    shift=offset(clock[0]+D(.5)*dt,amplitude,period)
    angle=D(2.)*D(micro.PI)*shift/D(PERIODIC_LENGTH)
    signed=base[j,f]+wp.cos(angle)*sine[j,f]-wp.sin(angle)*cosine[j,f]
    inward[j,f]=wp.max(-signed,D(0.))+mixing[j,f]
    outward[j,f]=wp.max(signed,D(0.))+mixing[j,f]

@wp.kernel
def transport_particles(state:wp.array(dtype=micro.State),clock:micro.F,dt:D,n:int,
                        wind:D,exchange:D,fall:D,amplitude:D,period:D):
    i=wp.tid();a=state[i];t=clock[0]
    if a.fate>=2 or a.birth>=t+dt:
        return
    h=dt
    if a.fate==0:
        a.fate=1
        h=t+dt-wp.max(a.birth,t)
    rho=D(917.) if a.phase==2 else D(1000.)
    drift=wp.vec2d(D(0.),fall*micro.terminal(micro.radius(a.mass,rho),rho))
    start=t+dt-h
    first_shift=wp.vec2d(offset(start,amplitude,period),D(0.))
    mid_shift=wp.vec2d(offset(start+D(.5)*h,amplitude,period),D(0.))
    v=micro.velocity(a.pos-first_shift,wind,D(0.))+micro.velocity(a.pos,D(0.),exchange)-drift
    midpoint=a.pos+D(.5)*h*v
    next_pos=a.pos+h*(micro.velocity(midpoint-mid_shift,wind,D(0.))+micro.velocity(midpoint,D(0.),exchange)-drift)
    x=next_pos[0]-D(PERIODIC_LENGTH)*wp.floor(next_pos[0]/D(PERIODIC_LENGTH))
    a.pos=wp.vec2d(x,next_pos[1])
    if next_pos[1]>=D(HEIGHT):
        a.fate=2
    elif next_pos[1]<D(0.):
        a.fate=3
    else:
        a.cell=micro.bin_id(a.pos,n)
    state[i]=a

@wp.kernel
def supply(vapor:M,added:micro.F,n:int,quantity:D):
    ix=wp.tid();j=ix*n
    vapor[j,0]+=quantity
    added[j]+=quantity


class Roll:
    def __init__(self,amplitude,period):
        self.amplitude=float(amplitude)
        self.period=float(period)

    def bind(self,c,grid,frame,device):
        n=c.n;k=2*np.pi/1600
        x,z=np.meshgrid(np.linspace(0,1600,n+1),np.linspace(0,600,n+1),indexing='ij')
        eta=z/600
        def faces(psi,internal):
            psi[-1]=psi[0];psi[:,0]=0.
            if internal:psi[:,-1]=0.
            fx=np.diff(psi,axis=1);fz=-np.diff(psi,axis=0)
            a=np.zeros((n,n,6));a[:,:,0]=-fx[:-1];a[:,:,1]=fx[1:]
            a[:,:,4]=-fz[:,:-1];a[:,:,5]=fz[:,1:]
            return c.dt*a.reshape(-1,6)
        base=faces(-np.sin(k*x)/k*c.entrain*eta**3,False)
        sine=faces(-np.sin(k*x)/k*c.eddy*np.sin(np.pi*eta),True)
        cosine=faces(-np.cos(k*x)/k*c.eddy*np.sin(np.pi*eta),True)
        np.testing.assert_allclose(base+sine,frame.transfer,atol=1e-10)
        for phi in np.linspace(-np.pi,np.pi,65):
            q=base+sine*np.cos(phi)-cosine*np.sin(phi)
            np.testing.assert_allclose(q.sum(1),0.,atol=1e-10)
            assert np.max((np.maximum(q,0)+frame.mixing).sum(1)/grid.volume)<.8
        self.data=[wp.array(np.ascontiguousarray(a),dtype=D,device=device) for a in (base,sine,cosine,frame.mixing)]
        self.device=device
        wp.load_module(module=surfaces.module,device=device)

    def faces(self,clock,dt,inward,outward):
        wp.launch(surfaces,inward.shape,inputs=[*self.data,inward,outward,clock,dt,self.amplitude,self.period],device=self.device)

    def move(self,state,clock,c):
        wp.launch(transport_particles,len(state),inputs=[state,clock,c.dt,c.n,c.eddy,c.entrain,c.fall,self.amplitude,self.period],device=self.device)
