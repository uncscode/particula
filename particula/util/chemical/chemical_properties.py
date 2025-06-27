from typing import Dict

from thermo.chemical import Chemical


def get_chemical_stp_properties(identifier: str) -> Dict[str, float]:
    """
    Return selected thermodynamic properties of *identifier* at STP.

    Parameters
    ----------
    identifier
        Any string accepted by ``thermo.Chemical`` (name, formula, CAS).

    Returns
    -------
    dict
        {
            "molar_mass": kg mol⁻¹,
            "density": kg m⁻³,
            "surface_tension": N m⁻¹,
            "pure_vapor_pressure": Pa,
        }
    """
    chem = Chemical(identifier)  # default T=298.15 K, P=1 atm
    # Ensure the state is STP in case caller changed defaults somewhere else

    return {
        "molar_mass": chem.MW / 1e3,  # kg/mol
        "density": chem.rho,  # kg/m³
        "surface_tension": chem.sigma,  # N/m
        "pure_vapor_pressure": chem.Psat,  # Pa
    }  # type: ignore
