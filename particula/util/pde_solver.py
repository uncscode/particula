""" PDE solvers for balance equation: ∂n/∂t + f(t,d,n) ∂n/∂d = g(t,d,n)
"""

class PreSolver:
    """ generic solver class
    """

    def __init__(self, **kwargs):
        """ set the physics
        """
        self.cond = kwargs.get("condensation", 0)
        self.coag = kwargs.get("coagulation", 0)
        self.nucl = kwargs.get("nucleation", 0)
        self.wall = kwargs.get("wall_loss", 0)
        self.dilu = kwargs.get("dilution", 0)

    def input_check(self, **kwargs):
        """ check the input:

            call input methods here to ensure units, etc.
        """


    def input_fix(self, **kwargs):
        """ fixing any problems with shapes, etc.
        """


class InitSolver(PreSolver):
    """ PDE solver for balance equation: ∂n/∂t + f(t,d,n) ∂n/∂d = g(t,d,n)
    """

    def __init__(self, **kwargs):
        """ set the physics
        """

        super().__init__(**kwargs)

        self.init_dist = kwargs.get("init_dist", 0)
        self.init_size = kwargs.get("init_size", 0)
        self.init_time = kwargs.get("init_time", 0)

    def input_check(self, **kwargs):
        """ check the input:

            call input methods here to ensure units, etc.
        """

    def set_steps(self, **kwargs):
        """ set the time step to avoid problems
        """
        return self.init_time / kwargs.get("steps", 0)

class LxWfSolver(InitSolver):
    """ Lax Wendroff's solver for balance equation:

        ∂n/∂t + f(t,d,n) ∂n/∂d = g(t,d,n)
    """

    def __init__(self, **kwargs):
        """ set the physics
        """

        super().__init__(**kwargs)

        self.time_span = kwargs.get("time_span", 0)
        self.time_step = self.set_steps

    def input_check(self, **kwargs):
        """ check the input:

            call input methods here to ensure units, etc.
        """

    def set_steps(self, **kwargs):
        """ set the time step to avoid problems
        """
