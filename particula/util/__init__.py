"""
Module for utility functions.
"""

# pylint: disable=unused-import
# flake8: noqa
# pyright: basic

from particula.util.arbitrary_round import (
    get_arbitrary_round,
)
from particula.util import constants
from particula.util.machine_limit import (
    get_safe_exp,
    get_safe_log,
    get_safe_log10,
)
from particula.util.reduced_quantity import (
    get_reduced_self_broadcast,
    get_reduced_value,
)
from particula.util.refractive_index_mixing import (
    get_effective_refractive_index,
)
from particula.util.converting.convert_dtypes import (
    get_coerced_type,
    get_dict_from_list,
    get_shape_check,
    get_values_of_dict,
)
from particula.util.converting.convert_kappa_volumes import (
    get_kappa_from_volumes,
    get_water_volume_from_kappa,
    get_solute_volume_from_kappa,
    get_water_volume_in_mixture,
)
from particula.util.converting.convert_mass_concentration import (
    get_volume_fraction_from_mass,
    get_mass_fraction_from_mass,
    get_mole_fraction_from_mass,
)
from particula.util.converting.convert_mole_fraction import (
    get_mass_fractions_from_moles,
)
from particula.util.converting.convert import (
    get_conversion_strategy,
    get_distribution_in_dn,
    get_pdf_distribution_in_pmf,
    SameScaleConversionStrategy,
    DNdlogDPtoPDFConversionStrategy,
    DNdlogDPtoPMFConversionStrategy,
    PMFtoPDFConversionStrategy,
)
from particula.util.converting.convert_units import (
    get_unit_conversion,
)
