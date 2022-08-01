""" 1st derivative, central differencing, accuracy up to 4

    borrowed from:
    https://github.com/ngam/hypersolver/blob/db89711fcdaae83ff8bf20857c381f4733680e39/hypersolver/accurate_derivative.py
"""

import numpy as np

weights = np.zeros((4, 4))
weights[0, -1:] = np.array([-1/2])
weights[1, -2:] = np.array([1/12, -2/3])
weights[2, -3:] = np.array([-1/60, 3/20, -3/4])
weights[3, -4:] = np.array([1/280, -4/105, 1/5, -4/5])


def acc4_derivative(_func, _xvar):
    """ 1st derivative, central differencing, accuracy up to 4
    """
    _nacc = 4
    derivative = np.zeros_like(_func/_xvar)*_func.u/_xvar.u
    derivative[0] = (_func[1] - _func[0])/(_xvar[1] - _xvar[0])
    derivative[-1] = (_func[-1] - _func[-2])/(_xvar[-1] - _xvar[-2])

    for nac, idx in zip(range(2, _nacc + 1, 2), range(_nacc//2)):

        derivative[nac//2] += (
            weights[nac//2-1, -idx-1]*(_func[0] - _func[nac]) /
            ((_xvar[nac] - _xvar[0]) / float(nac)))

        derivative[-nac//2-1] += (
            weights[nac//2-1, -idx-1]*(_func[-nac-1] - _func[-1]) /
            ((_xvar[-1] - _xvar[-nac-1]) / float(nac)))

        derivative[nac//2+1:-nac//2-1] += (
            weights[_nacc//2-1, -idx-1]*(_func[1:-nac-1] - _func[nac+1:-1]) /
            ((_xvar[nac+1:-1] - _xvar[1:-nac-1]) / float(nac)))

    return derivative
