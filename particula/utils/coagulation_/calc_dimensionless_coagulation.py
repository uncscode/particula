""" A class with methods for dimensionless coagulation
"""

import numpy as np

# from particula import u
from particula.utils.particle_ import (
    diffusive_knudsen,
)


class DimensionlessCoagulation:

    """ dimensionless coagulation
    """

    def __init__(
        self,
        radius, other_radius,
        density=1000, other_density=1000,
        charge=0, other_charge=0,
        temperature=298,
        mfp_air=66.4e-9,
        dyn_vis_air=1.8e-05,
    ) -> float:

        """ Dimensionless particle--particle coagulation kernel.

            Attributes:
                radius        (float) [m]
                other_radius  (float) [m]
                density       (float) [kg/m^3]        (default: 1000)
                other_density (float) [kg/m^3]        (default: 1000)
                charge        (int)   [dimensionless] (default: 0)
                other_charge  (int)   [dimensionless] (default: 0)
                temperature   (float) [K]             (default: 298)
                mfp_air       (float) [m]             (default: 66.4e-9)
                dyn_vis_air   (float) [kg/m/s]        (default: 1.8e-05)
        """

        self.radius = radius
        self.other_radius = other_radius
        self.density = density
        self.other_density = other_density
        self.charge = charge
        self.other_charge = other_charge
        self.temperature = temperature
        self.mfp_air = mfp_air
        self.dyn_vis_air = dyn_vis_air

    def hard_sphere(self):

        """ Dimensionless particle--particle coagulation kernel.
        """

        HS_CS = [25.836, 11.211, 3.502, 7.211]
        difkn = diffusive_knudsen(
            self.radius, self.other_radius,
            self.density, self.other_density,
            self.charge, self.other_charge,
            self.temperature,
            self.mfp_air,
            self.dyn_vis_air,
        )

        upstairs = (
            (4 * np.pi * difkn**2) +
            (HS_CS[0] * difkn**3) +
            ((8 * np.pi)**(1/2) * HS_CS[1] * difkn**4)
        )

        downstairs = (
            1 +
            (HS_CS[2] * difkn) +
            (HS_CS[3] * difkn**2) +
            (HS_CS[1] * difkn**3)
        )

        return upstairs / downstairs
