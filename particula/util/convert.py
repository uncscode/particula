"""conversion functions common for aerosol processing
"""

from typing import Union, Tuple, Any, List, Dict
from numpy.typing import NDArray
import numpy as np


def coerce_type(data, dtype):
    """
    Coerces data to dtype if it is not already of that type.

    Example:
        >>> coerce_type(1, float)
        1.0
        >>> coerce_type([1, 2, 3], np.ndarray)
        array([1, 2, 3])

    Args:
        data: The data to be coerced.
        dtype: The desired data type.

    Returns:
        The coerced data.

    Raises:
        ValueError: If the data cannot be coerced to the desired type.
    """
    if not isinstance(data, dtype):
        try:
            return np.array(data) if dtype == np.ndarray else dtype(data)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not coerce {data} to {dtype}") from exc
    return data


def round_arbitrary(
    values: Union[float, list[float], np.ndarray],
    base: Union[float, np.float64] = 1.0,
    mode: str = "round",
    nonzero_edge: bool = False,
) -> Union[float, NDArray[np.float64]]:
    """
    Rounds the input values to the nearest multiple of the base.

    For values exactly halfway between rounded decimal values, "Bankers
    rounding applies" rounds to the nearest even value. Thus 1.5 and 2.5
    round to 2.0, -0.5 and 0.5 round to 0.0, etc.

    Args:
        values: The values to be rounded.
        base: The base to which the values should be rounded.
        mode: The rounding mode: 'round', 'floor', 'ceil'
        nonzero_edge: If true the zero values are replaced
        by the original values.

    Returns:
        rounded: The rounded values.
    """
    # Check if values is a NumPy array
    working_values = coerce_type(values, np.ndarray)
    base = coerce_type(base, float)

    # Validate base parameter
    if not isinstance(base, float) or base <= 0:
        raise ValueError("base must be a positive float")
    # Validate mode parameter
    if mode not in ["round", "floor", "ceil"]:
        raise ValueError("mode must be one of ['round', 'floor', 'ceil']")

    # Calculate rounding factors
    factor = np.array([-0.5, 0, 0.5])

    # Compute rounded values
    rounded = base * np.round(
        working_values / base
        + factor[np.array(["floor", "round", "ceil"]).tolist().index(mode)]
    )

    # Apply round_nonzero mode
    if nonzero_edge:
        rounded = np.where(rounded != 0, rounded, working_values)

    return float(rounded) if isinstance(values, float) else rounded


def radius_diameter(value: float, to_diameter: bool = True) -> float:
    """
    Convert a radius to a diameter, or vice versa.

    Args:
        value: The value to be converted.
        to_diameter: If True, convert from radius to diameter.
        If False, convert from diameter to radius.

    Returns:
        The converted value.
    """
    return value * 2 if to_diameter else value / 2


def volume_to_length(
    volume: Union[float, NDArray[np.float64]], length_type: str = "radius"
) -> Union[float, NDArray[np.float64]]:
    """
    Convert a volume to a radius or diameter.

    Args:
        volume: The volume to be converted.
        length_type: The type of length to convert to ('radius' or 'diameter')
        Default is 'radius'.

    Returns:
        The converted length.
    """

    if length_type not in ["radius", "diameter"]:
        raise ValueError("length_type must be radius or diameter")

    radius = (volume * 3 / (4 * np.pi)) ** (1 / 3)

    return radius if length_type == "radius" else radius * 2


def length_to_volume(
    length: Union[float, np.ndarray], length_type: str = "radius"
) -> Union[float, np.ndarray]:
    """
    Convert radius or diameter to volume.

    Args:

        length: The length to be converted.
        length_type: The type of length ('radius' or 'diameter').
            Default is 'radius'.

    Returns:
    --------
        The volume.
    """
    if length_type == "diameter":
        length = length / 2
    elif length_type != "radius":
        raise ValueError("length_type must be radius or diameter")
    return (4 / 3) * np.pi * (length**3)


def kappa_volume_solute(
    volume_total: Union[float, np.ndarray],
    kappa: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the volume of solute in a volume of total solution,
    given the kappa parameter and water activity.

    Args:

        volume_total: The volume of the total solution.
        kappa: The kappa parameter.
        water_activity: The water activity.

    Returns:
    --------
        The volume of solute as a numpy array.
    """

    kappa = max(kappa, 1e-16)  # Avoid division by zero
    if water_activity <= 1e-16:  # early return for low water activity
        return volume_total

    vol_factor = (water_activity - 1) / (
        water_activity * (1 - kappa - 1 / water_activity)
    )
    return volume_total * np.array(vol_factor)


def kappa_volume_water(
    volume_solute: Union[float, NDArray[np.float64]],
    kappa: Union[float, NDArray[np.float64]],
    water_activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the volume of water given volume of solute, kappa parameter,
    and water activity.

    Args:

        volume_solute: The volume of solute.
        kappa: The kappa parameter.
        water_activity: The water activity.

    Returns:
    --------
        The volume of water as a float.
    """
    # Avoid division by zero
    water_activity = min(water_activity, 1 - 1e-16)

    if water_activity <= 1e-16:  # early return for low water activity
        return volume_solute * 0

    return volume_solute * kappa / (1 / water_activity - 1)


def kappa_from_volume(
    volume_solute: Union[float, np.ndarray],
    volume_water: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the kappa parameter from the volume of solute and water,
    given the water activity.

    Args:

        volume_solute: The volume of solute.
        volume_water: The volume of water.
        water_activity: The water activity.

    Returns:
    --------
        The kappa parameter as a float.
    """
    # Avoid division by zero
    water_activity = np.where(
        water_activity > 1 - 1e-16, 1 - 1e-16, water_activity
    )

    return (1 / water_activity - 1) * volume_water / volume_solute


def mole_fraction_to_mass_fraction(
    mole_fraction0: float, molecular_weight0: float, molecular_weight1: float
) -> Tuple[float, float]:
    """
    Convert mole fraction to mass fraction.

    Args:
        mole_fraction0: The mole fraction of the first component.
        molecular_weight0: The molecular weight of the first component.
        molecular_weight1: The molecular weight of the second component.

    Returns:
        A tuple containing the mass fractions of the two components as floats.
    """
    mass_fraction0 = (
        mole_fraction0
        * molecular_weight0
        / (
            mole_fraction0 * molecular_weight0
            + (1 - mole_fraction0) * molecular_weight1
        )
    )
    mass_fraction1 = 1 - mass_fraction0
    return mass_fraction0, mass_fraction1


def mole_fraction_to_mass_fraction_multi(
    mole_fractions: list[float], molecular_weights: list[float]
) -> list[float]:
    """Convert mole fractions to mass fractions for N components.
    Assumes that sum(mole_fractions) == 1.

    Args:
        mole_fractions: A list of mole fractions.
        molecular_weights: A list of molecular weights.

    Returns:
        A list of mass fractions.
    """
    if np.sum(mole_fractions) != 1:
        raise ValueError("Sum of mole fractions must be 1")

    total_molecular_weight = np.sum(
        [mf * mw for mf, mw in zip(mole_fractions, molecular_weights)]
    )
    return [
        mf * mw / total_molecular_weight
        for mf, mw in zip(mole_fractions, molecular_weights)
    ]


def mass_concentration_to_mole_fraction(
    mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Convert mass concentrations to mole fractions for N components.

    Args:
        mass_concentrations: A list or ndarray of mass concentrations
        (e.g., kg/m^3).
        molar_masses: A list or ndarray of molecular weights (e.g., g/mol).

    Returns:
        An ndarray of mole fractions.

    Note:
        The mole fraction of a component is given by the ratio of its molar
        concentration to the total molar concentration of all components.
    """
    # Convert mass concentrations to moles for each component
    moles = mass_concentrations / molar_masses

    # Calculate total moles in the mixture
    total_moles = np.sum(moles)

    return moles / total_moles


def mass_concentration_to_volume_fraction(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert mass concentrations to volume fractions for N components.

    Args:
        mass_concentrations: A list or ndarray of mass concentrations
        (e.g., kg/m^3).
        densities: A list or ndarray of densities of each component
        (e.g., kg/m^3).

    Returns:
        An ndarray of volume fractions.

    Note:
        The volume fraction of a component is calculated by dividing the volume
        of that component (derived from mass concentration and density) by the
        total volume of all components.
    """
    # Calculate volumes for each component using mass concentration and density
    volumes = mass_concentrations / densities  # [:, np.newaxis]

    # Calculate total volume of the mixture
    total_volume = np.sum(volumes)  # axis=0

    # Calculate volume fractions by dividing the volume of each component by
    # the total volume

    return volumes / total_volume


def mass_fraction_to_volume_fraction(
    mass_fraction: float, density_solute: float, density_solvent: float
) -> Tuple[float, float]:
    """
    Converts the mass fraction of a solute to the volume fraction in a
    binary mixture.

    Args:
        mass_fraction (float): The mass fraction of the solute in the mixture.
        density_solute (float): The density of the solute.
        density_solvent (float): The density of the solvent.

    Returns:
        Tuple[float, float]: A tuple containing the volume fraction of the
            solute and solvent in the mixture.

    Example:
        If `mass_fraction` is 0.5, `density_solute` is 1.5 g/cm^3, and
        `density_solvent` is 2 g/cm^3, this function returns (0.5714, 0.4285),
        indicating that the solute and solvent occupy 57% and 42% of the
        mixture's volume, respectively.
    """
    volume_fraction_solute = (
        mass_fraction
        / density_solute
        / (
            mass_fraction / density_solute
            + (1 - mass_fraction) / density_solvent
        )
    )
    volume_fraction_solvent = 1 - volume_fraction_solute
    return volume_fraction_solute, volume_fraction_solvent


def volume_water_from_volume_fraction(
    volume_solute_dry: float, volume_fraction_water: float
) -> float:
    """
    Calculates the volume of water in a volume of solute, given the volume
    fraction of water in the mixture.

    Args:
        volume_solute_dry (float): The volume of the solute, excluding water.
        volume_fraction_water (float): The volume fraction of water in the
                            mixture, expressed as a decimal between 0 and 1.

    Returns:
        float: The volume of water in the mixture, in the same units as
            `volume_solute_dry`.

    Example:
        If `volume_solute_dry` is 100 mL and `volume_fraction_water` is 0.8,
        this function returns 400 mL, indicating that there are 400 mL of water
        in the total 100 mL + 400 mL mixture.
    """
    return (
        volume_fraction_water * volume_solute_dry / (1 - volume_fraction_water)
    )


def effective_refractive_index(
    m_zero: Union[float, complex],
    m_one: Union[float, complex],
    volume_zero: float,
    volume_one: float,
) -> Union[float, complex]:
    """
    Calculate the effective refractive index of a mixture of two solutes, given
    the refractive index of each solute and the volume of each solute. The
    mixing is based on volume-weighted molar refraction.

    Args:
        m_zero: The refractive index of solute 0.
        m_one: The refractive index of solute 1.
        volume_zero: The volume of solute 0.
        volume_one: The volume of solute 1.

    Returns:
        The effective refractive index of the mixture.

    References:
        Liu, Y., &#38; Daum, P. H. (2008). Relationship of refractive
        index to mass density and self-consistency mixing rules for
        multicomponent mixtures like ambient aerosols.
        Journal of Aerosol Science, 39(11), 974-986.
        https://doi.org/10.1016/j.jaerosci.2008.06.006
    """
    volume_total = volume_zero + volume_one
    r_effective = volume_zero / volume_total * (m_zero - 1) / (
        m_zero + 2
    ) + volume_one / volume_total * (m_one - 1) / (
        m_one + 2
    )  # molar refraction mixing

    # convert to refractive index
    return (2 * r_effective + 1) / (1 - r_effective)


def convert_sizer_dn(
    diameter: np.ndarray, dn_dlogdp: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """
    Converts the sizer data from dn/dlogdp to d_num.

    The bin width is defined as the  difference between the upper and lower
    diameter limits of each bin. This function calculates the bin widths
    based on the input diameter array. Assumes a log10 scale for dp edges.

    Args:
        diameter (np.ndarray): Array of particle diameters.
        dn_dlogdp (np.ndarray): Array of number concentration of particles per
        unit logarithmic diameter.
        inverse (bool): If True, converts from d_num to dn/dlogdp.

    Returns:
        np.ndarray: Array of number concentration of particles
        per unit diameter.

    References:
        Eq: dN/dlogD_p = dN/( log(D_{p-upper}) - log(D_{p-lower}) )
        https://tsi.com/getmedia/1621329b-f410-4dce-992b-e21e1584481a/
        PR-001-RevA_Aerosol-Statistics-AppNote?ext=.pdf
    """
    assert len(diameter) > 0, "Inputs must be non-empty arrays."
    # Compute the bin widths
    delta = np.zeros_like(diameter)
    delta[:-1] = np.diff(diameter)
    delta[-1] = delta[-2] ** 2 / delta[-3]

    # Compute the lower and upper bin edges
    lower = diameter - delta / 2
    upper = diameter + delta / 2

    # if dn_dlogdp.ndim == 2:
    #     # expand diameter by one dimension so it can be broadcast
    #     lower = np.expand_dims(lower, axis=1)
    #     upper = np.expand_dims(upper, axis=1)
    if inverse:
        # Convert from dn to dn/dlogdp
        return dn_dlogdp / np.log10(upper / lower)

    return dn_dlogdp * np.log10(upper / lower)


def list_to_dict(list_of_str: list) -> dict:
    """
    Converts a list of strings to a dictionary. The keys are the strings
    and the values are the index of the string in the list.

    Args:
        list_of_str (list): A non-empty list of strings.

    Returns:
        dict: A dictionary where the keys are the strings and the values are
            the index of the string in the list.
    """
    assert all(list_of_str), "Input list_of_str must not be empty."
    assert all(
        isinstance(item, str) for item in list_of_str
    ), "Input \
        list_of_str must contain only strings."

    # Create a dictionary from the list of strings using a dictionary
    # comprehension
    return {str_val: i for i, str_val in enumerate(list_of_str)}


def get_values_in_dict(
    key_list: List[str], dict_to_check: Dict[str, Any]
) -> List[Any]:
    """
    Returns a list of values for keys in a dictionary.

    Example:
    >>> my_dict = {'a': 1, 'b': 2, 'c': 3}
    >>> get_values_in_dict(['a', 'c'], my_dict)
    [1, 3]

    Args:
        key_list: List of keys to check in the dictionary.
        dict_to_check: The dictionary to check for the given keys.

    Returns:
        List: A list of values for keys in the dictionary.

    Raises:
        KeyError: If any of the keys in the `key_list` are not present in
        the dictionary.
    """
    values = []
    for key in key_list:
        if key in dict_to_check:
            values.append(dict_to_check[key])
        else:
            raise KeyError(
                f"Key '{key}' not found in the dictionary. Available keys:"
                + f"{list(dict_to_check.keys())}"
            )
    return values


def data_shape_check(
    time: np.ndarray,
    data: np.ndarray,
    header: list,
) -> np.ndarray:
    """
    Check the shape of the input data and header list, and reshape the data if
    necessary. The data array can be 1D or 2D. If the data array is 2D, the
    time array must match the last dimensions of the data array. If the data
    array is 1D, the header list must be a single entry.

    Args:
        time (np.ndarray): 1D array of time values.
        data (np.ndarray): 1D or 2D array of data values.
        header (list): List of header values.

    Returns:
        Reshaped data array.

    Raises:
        ValueError: If the length of the header list does not match the first
        dimension of the data array.
    """

    # Check if data_new is 2D or 1D
    if len(data.shape) == 2:
        # Check if time matches the dimensions of data
        if len(time) == data.shape[0] and len(time) == data.shape[1]:
            concatenate_axis_new = 1  # Default to the axis=1
        else:
            # Find the axis that doesn't match the length of time
            concatenate_axis_new = np.argwhere(
                np.array(data.shape) != len(time)
            ).flatten()[0]
        # Reshape new data so the concatenate axis is axis=1
        data = np.moveaxis(data, concatenate_axis_new, 1)

        # check header list length matches data_new shape
        if len(header) != data.shape[1]:
            print(
                f"header len: {len(header)} vs. data.shape: \
                  {data.shape}"
            )
            print(header)
            raise ValueError(
                "Header list length must match the second \
                              dimension of data_new."
            )
    elif len(header) == 1:
        # Reshape new data so the concatenate axis is axis=1
        data = np.expand_dims(data, 1)

    else:
        raise ValueError(
            "Header list must be a single entry if data_new \
                              is 1D."
        )
    return data


def distribution_convert_pdf_pms(
    x_array: np.ndarray, distribution: np.ndarray, to_pdf: bool = True
) -> np.ndarray:
    """
    Convert between a probability density function (PDF) and a probability
    mass spectrum (PMS) based on the specified direction.

    Args:
        x_array : An array of radii corresponding to the bins of the
            distribution, shape (m).
        distribution : The concentration values of the distribution
            (either PDF or PMS) at the given radii. Supports broadcasting
            across x_array (n,m).
        to_PDF : Direction of conversion. If True, converts PMS to PDF.
            If False, converts PDF to PMS.

    Returns:
        converted_distribution : The converted distribution array
            (either PDF or PMS).
    """

    # Calculate the differences between consecutive x_array values for bin
    # widths.
    delta_x_array = np.empty_like(x_array)
    delta_x_array[:-1] = np.diff(x_array)
    # For the last bin, extrapolate the width assuming constant growth
    # rate from the last two bins.
    delta_x_array[-1] = delta_x_array[-2] ** 2 / delta_x_array[-3]

    # Converting PMS to PDF by dividing the PMS values by the bin widths. or
    # Converting PDF to PMS by multiplying the PDF values by the bin widths.
    return (
        distribution / delta_x_array
        if to_pdf
        else distribution * delta_x_array
    )
