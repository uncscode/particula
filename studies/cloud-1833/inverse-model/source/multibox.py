"""Run independent frozen geometries concurrently on the same CUDA device."""
from forward import *
def many(n,seg,jobs):
    prepared=[];boxes=[];params=[]
    for k,en,two,tau,p in jobs:
        ix,a,mapper,arrays=prepare(n,k,seg,en)
        prepared.append((mapper,arrays));boxes.append([v.ptr for v in arrays]+[two,en,tau,0]);params.append(p)
    descriptors=gpu(np.array(boxes,dtype=np.uint64));pars=gpu(np.array(params))
    out=wp.empty((len(jobs),len(ix),n+NC+4),dtype=wp.float64,device=dev)
    lib=ctypes.CDLL(os.environ.get('MULTI_SOLVER',str(Path(os.environ['INVERSE_SOLVER']).with_name('multi.so'))));fn=lib.many_run
    fn.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*2+[ctypes.c_void_p]*3
    wp.synchronize_device(dev)
    status=fn(len(jobs),n,len(ix),a['dx'],6 if n==288 else 12,descriptors.ptr,pars.ptr,out.ptr)
    if status:raise RuntimeError(f'CUDA batch returned {status}')
    res=out.numpy()
    if not np.isfinite(res).all() or res.min() < -1e-7:raise ValueError(('invalid transfer trajectory',n,seg,res.min()))
    predictions=[]
    for state,(mapper,arrays) in zip(res,prepared):
        np.testing.assert_allclose(state[0,:n],arrays[1].numpy(),rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(state[0,n+NC],frame.measured_cal.to_numpy()[ix[0]],rtol=1e-12)
        predictions.append((state[:,:n+NC]@mapper.T,state[:,n+NC:].copy()))
    return predictions
