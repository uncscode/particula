"""Molar mass conversions."""


def to_molar_mass_ratio(molar_mass, other_molar_mass=18.01528):
    """
    Convert the given molar mass to a molar mass ratio with respect to water.
    (MW water / MW organic)

    Args:
    molar_mass (np.array): The molar mass of the organic compound.
    other_molar_mass (float, optional): The molar mass of the other compound.
        Defaults to 18.01528.

    Returns:
    np.array: The molar mass ratio with respect to water.
    """
    if isinstance(molar_mass, list):
        return [other_molar_mass / mm for mm in molar_mass]
    return other_molar_mass / molar_mass


def from_molar_mass_ratio(molar_mass_ratio, other_molar_mass=18.01528):
    """
    Convert the given molar mass ratio (MW water / MW organic) to a
    molar mass with respect to the other compound.

    Args:
    molar_mass_ratio (np.array): The molar mass ratio with respect to water.
    other_molar_mass (float, optional): The molar mass of the other compound.
        Defaults to 18.01528.

    Returns:
    np.array: The molar mass of the organic compound.
    """
    if isinstance(molar_mass_ratio, list):
        return [other_molar_mass * mm for mm in molar_mass_ratio]
    return other_molar_mass * molar_mass_ratio
