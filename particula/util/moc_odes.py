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

from scipy.integrate import odeint
from particula.util.coagulation_rate import CoagulationRate


class SolveODE:
    """ a class to solve the ODE:

        Need:
        1. initial distribution     nums_init
        2. initial radius           rads_init
        3. time span                time_span

        Also:
        1. coagulation kernel       coag_kern
    """
    def __init__(self, **kwargs):
        """ constructor
        """
        self.nums_init = kwargs.get("nums_init", None)
        self.rads_init = kwargs.get("rads_init", None)
        self.time_span = kwargs.get("time_span", None)
        self.kwargs = kwargs

    def ode_func(self, _nums, _, _rads, _phys):
        """ function to integrate
        """

        coag = CoagulationRate(
            distribution=_nums,
            radius=_rads,
            **_phys.kwargs
        )

        return coag.coag_gain().m - coag.coag_loss().m

    def solve_ode(self):
        """ utilize scipy.integrate.odeint
        """

        def _func(_nums, _, _rads):
            """ internal function to integrate

                Here:
                1. _nums: the distribution as it evolves
                2. _    : the time as it evolves
                3. _rads: the radius as it evolves
                4. _coag: the coagulation kernel

                Note: it is okay that _ (_time) is not used below.
            """

            vals = CoagulationRate(
                distribution=_nums,
                radius=_rads,
                kernel=_coag,
            )

            return vals.coag_gain().m - vals.coag_loss().m

        return odeint(
            _func,
            self.nums_init,
            self.time_span,
        )
