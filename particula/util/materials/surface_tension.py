from typing import Union
import numpy as np
from numpy.typing import NDArray
from thermo.chemical import Chemical


def get_surface_tension(chemical_identifier: str,
                        temperature: Union[float, NDArray[np.float64]]
                        ) -> NDArray[np.float64]:
    """
    Return surface tension [N/m] of *chemical_identifier* at the requested
    temperature(s).

    Parameters
    ----------
    chemical_identifier : str
        Any string accepted by thermo.chemical.Chemical.
    temperature : float | ndarray[float64]
        Temperature(s) in Kelvin.

    Returns
    -------
    ndarray[float64]  (scalar returned as 0-d array)
    """
    temps = np.asarray(temperature, dtype=np.float64)
    chem = Chemical(chemical_identifier)

    st = np.vectorize(lambda T: chem.SurfaceTension(T=T), otypes=[np.float64])(temps)
    return st
