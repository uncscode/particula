"""Coagulation functions exposed via __init__.

If you want specific sub functions, import them directly from
`particula.next.dynamics.coagulation.xxx.xxx` instead.
"""

# pylint: disable=unused-import, disable=line-too-long
# flake8: noqa
# pyright: basic

from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel,
    get_brownian_kernel_via_system_state,
)
from particula.dynamics.coagulation.charged_kernel_strategy import (
    HardSphereKernelStrategy,
    CoulombDyachkov2007KernelStrategy,
    CoulombGatti2008KernelStrategy,
    CoulombGopalakrishnan2012KernelStrategy,
    CoulumbChahl2019KernelStrategy,
)
from particula.dynamics.coagulation import coagulation_strategy
from particula.dynamics.coagulation.coagulation_rate import (
    get_coagulation_loss_rate_continuous,
    get_coagulation_loss_rate_discrete,
    get_coagulation_gain_rate_discrete,
    get_coagulation_gain_rate_continuous,
)
from particula.dynamics.coagulation.charged_dimensionless_kernel import (
    get_hard_sphere_kernel,
    get_coulomb_kernel_dyachkov2007,
    get_coulomb_kernel_gatti2008,
    get_coulomb_kernel_gopalakrishnan2012,
    get_coulomb_kernel_chahl2019,
)
from particula.dynamics.coagulation.turbulent_shear_kernel import (
    get_turbulent_shear_kernel_st1956_via_system_state,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.turbulent_dns_kernel_ao2008 import (
    get_turbulent_dns_kernel_ao2008,
    get_turbulent_dns_kernel_ao2008_via_system_state,
)
from particula.dynamics.coagulation.coagulation_factories import (
    CoagulationFactory,
)