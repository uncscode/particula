"""Get selected thermodynamic properties of a chemical at STP."""

from typing import Dict

from .thermo_import import Chemical


def get_chemical_stp_properties(identifier: str) -> Dict[str, float]:
    """Return selected thermodynamic properties of *identifier* at STP.

    Parameters
    ----------
    identifier
        Any string accepted by ``thermo.Chemical`` (name, formula, CAS).

    Returns:
    -------
    dict
        {
            "molar_mass": kg mol⁻¹,
            "density": kg m⁻³,
            "surface_tension": N m⁻¹,
            "pure_vapor_pressure": Pa,
        }
    """
    if Chemical is None:
        raise ImportError(
            "The 'thermo' package is required. "
            "Please install it using 'pip install thermo'."
        )

    chem = Chemical(
        identifier, T=298.15, P=101325
    )  # Explicitly set STP: T=298.15 K, P=101325 Pa

    return {
        "molar_mass": chem.MW / 1e3,  # kg/mol
        "density": chem.rho,  # kg/m³
        "surface_tension": chem.sigma,  # N/m
        "pure_vapor_pressure": chem.Psat,  # Pa
        "cas_number": chem.CAS,  # CAS number
        "name": chem.name,  # Chemical name
        "smiles": chem.smiles,  # SMILES representation
        "formula": chem.formula,  # Chemical formula
        "pubchem_id": chem.PubChem,  # PubChem ID
    }  # type: ignore
