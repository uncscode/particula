""" method-of-characteristics ODEs

    For quasilinear PDEs of the form:

    ∂n/∂t + f(t,d,n) ∂n/∂d = g(t,d,n)

    initial: n(0,d) = n0(d)

    dt / 1 = dd / f = dn / g

    or

    dn / dt = g / 1 AND dn / dd = g / f
    n(0) = n0
    d(0) = d0

    or the parameteric form:

    t = t(s), d = d(s), n = n(s)
    n(0,d) = n0(d) --> n(t=0, d=d0) = n0(d0)
    t(0) = 0
    d(0) = d0
    n(0) = n0

    dt/1 = dd/f = dn/g = ds

    dt/ds = 1 --> t = s + constant = s
    dd/ds = f --> dd/dt = f(t,d,n) --> d(t,d,n)
    dn/ds = g --> dn/dt = g(t,d,n) --> g(t,d,n)

    MEANING: simultaneously solve
    1. d(radius)/d(time) = f(radius,time,denisty) with radius(0) = d0
    2. d(density)/d(time) = g(radius,time,density) with density(0) = n0(d0)
"""

# def solver(f, g, d0, n0, t0, dt):
#     """ solve it!
#     """
#     pass
