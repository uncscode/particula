from typing import Union
import numpy as np
from numpy.typing import NDArray
from thermo.chemical import Chemical


def get_vapor_pressure(chemical_identifier: str,
                       temperature: Union[float, NDArray[np.float64]]
                       ) -> NDArray[np.float64]:
    """
    Return saturation-vapor pressure [Pa] of *chemical_identifier* at the
    requested temperature(s).

    Parameters
    ----------
    chemical_identifier : str
        Any string accepted by thermo.chemical.Chemical (name, CAS, formula…).
    temperature : float | ndarray[float64]
        Temperature(s) in Kelvin.

    Returns
    -------
    ndarray[float64]  (scalar returned as 0-d array)
    """
    temps = np.asarray(temperature, dtype=np.float64)
    chem = Chemical(chemical_identifier)

    # Vectorised call to thermo correlation
    vp = np.vectorize(lambda T: chem.VaporPressure(T=T), otypes=[np.float64])(temps)
    return vp
