"""Module for gas changes in thermodynamic processes."""


from typing import Union
from numpy.typing import NDArray
import numpy as np
from particula.next.runable import Runnable
from particula.next.aerosol import Aerosol


def adiabatic_pressure_change(
            temperature_initial: float,
            pressure_initial: float,
            pressure_final: float,
            gamma: float = 1.4
        ) -> float:
    """
    Calculates the final temperature of a gas undergoing an adiabatic change
    due to a pressure step.

    Args:
    -----
    - temperature_initial (float): The initial temperature of the gas in
    Kelvin.
    - pressure_initial (float): The initial pressure of the gas in Pascals.
    - pressure_final (float): The final pressure of the gas in Pascals.
    - gamma (float): The adiabatic index of the gas, Heat capacity ratio of
    the gas (Cp/Cv). Default is 1.4.

    Returns:
    --------
    - temperature_final (float): The final temperature of the gas in Kelvin.

    References:
    -----------
    - https://en.wikipedia.org/wiki/Adiabatic_process
    """
    temperature_final = temperature_initial * (
        pressure_final / pressure_initial) ** ((gamma - 1) / gamma)
    return temperature_final


class AdiabaticPressureChange(Runnable):
    """
    A class for running an adiabatic pressure change process.
    """

    def __init__(
        self,
        aerosol: Aerosol,
        new_pressure: float,
        gamma: float = 1.4
    ):
        self.aerosol = aerosol
        self.gamma = gamma
        self.new_pressure = new_pressure

    def execute(self, aerosol: Aerosol) -> Aerosol:
        """
        Execute the adiabatic pressure change process.

        Parameters:
        - aerosol (Aerosol): The aerosol instance to modify.

        Returns:
        - aerosol (Aerosol): The modified aerosol instance.
        """
        for gas in aerosol.iterate_gas():
            gas.temperature = adiabatic_pressure_change(
                gas.temperature,
                gas.total_pressure,
                self.new_pressure,
                self.gamma)
        return aerosol

    def rate(self, aerosol: Aerosol) -> Union[float, NDArray[np.float_]]:
        """
        Return the rate of the adiabatic pressure change process.

        Parameters:
        - aerosol (Aerosol): The aerosol instance to modify.

        Returns:
        - rate (float): The rate of the process.
        """
        return self.new_pressure - np.array([
            gas.total_pressure for gas in aerosol.iterate_gas()])


# def adiabatic_volume_change(
#             temperature_initial: Union[float, NDArray[np.float_]],
#             volume_initial: Union[float, NDArray[np.float_]],
#             volume_final: Union[float, NDArray[np.float_]],
#             gamma: Union[float, NDArray[np.float_]] = 1.4
#         ):
#     """
#     Calculates the final temperature of a gas undergoing an adiabatic change
#     due to a volume step.

#     Args:
#     -----
#     - temperature_initial (float): The initial temperature of the gas in
#     Kelvin.
#     - volume_initial (float): The initial volume of the gas in m^3.
#     - volume_final (float): The final volume of the gas in m^3.
#     - gamma (float): The adiabatic index of the gas, Heat capacity ratio of
#     the gas (Cp/Cv). Default is 1.4.

#     Returns:
#     --------
#     - temperature_final (float): The final temperature of the gas in Kelvin.

#     References:
#     -----------
#     - https://en.wikipedia.org/wiki/Adiabatic_process
#     """
#     temperature_final = temperature_initial * (
#         volume_initial / volume_final) ** (gamma - 1)
#     return temperature_final


# def isothermal_pressure_change(
#             pressure_initial: Union[float, NDArray[np.float_]],
#             volume_initial: Union[float, NDArray[np.float_]],
#             volume_final: Union[float, NDArray[np.float_]]
#         ):
#     """
#     Calculates the final pressure of a gas undergoing an isothermal change
#     due to a volume step.

#     Args:
#     -----
#     - pressure_initial (float): The initial pressure of the gas in Pascals.
#     - volume_initial (float): The initial volume of the gas in m^3.
#     - volume_final (float): The final volume of the gas in m^3.

#     Returns:
#     --------
#     - pressure_final (float): The final pressure of the gas in Pascals.

#     References:
#     -----------
#     - https://en.wikipedia.org/wiki/Isothermal_process
#     """
#     pressure_final = pressure_initial * volume_initial / volume_final
#     return pressure_final


# def isothermal_volume_change(
#             pressure_initial: Union[float, NDArray[np.float_]],
#             volume_initial: Union[float, NDArray[np.float_]],
#             pressure_final: Union[float, NDArray[np.float_]]
#         ):
#     """
#     Calculates the final volume of a gas undergoing an isothermal change
#     due to a pressure step.

#     Args:
#     -----
#     - pressure_initial (float): The initial pressure of the gas in Pascals.
#     - volume_initial (float): The initial volume of the gas in m^3.
#     - pressure_final (float): The final pressure of the gas in Pascals.

#     Returns:
#     --------
#     - volume_final (float): The final volume of the gas in m^3.

#     References:
#     -----------
#     - https://en.wikipedia.org/wiki/Isothermal_process
#     """
#     volume_final = volume_initial * pressure_initial / pressure_final
#     return volume_final


# def isobaric_temperature_change(
#             temperature_initial: Union[float, NDArray[np.float_]],
#             volume_initial: Union[float, NDArray[np.float_]],
#             volume_final: Union[float, NDArray[np.float_]],
#             gamma: Union[float, NDArray[np.float_]] = 1.4
#         ) -> Union[float, NDArray[np.float_]]:
#     """
#     Calculates the final temperature of a gas undergoing an isobaric change
#     due to a volume step.

#     Args:
#     -----
#     - temperature_initial (float): The initial temperature of the gas in
#     Kelvin.
#     - volume_initial (float): The initial volume of the gas in m^3.
#     - volume_final (float): The final volume of the gas in m^3.
#     - gamma (float): The adiabatic index of the gas, Heat capacity ratio of
#     the gas (Cp/Cv). Default is 1.4.

#     Returns:
#     --------
#     - temperature_final (float): The final temperature of the gas in Kelvin.

#     References:
#     -----------
#     - https://en.wikipedia.org/wiki/Isobaric_process
#     """
#     temperature_final = temperature_initial * (
#         volume_final / volume_initial) ** (gamma)
#     return temperature_final


# def isochoric_temperature_change(
#             temperature_initial: Union[float, NDArray[np.float_]],
#             pressure_initial: Union[float, NDArray[np.float_]],
#             pressure_final: Union[float, NDArray[np.float_]],
#             gamma: Union[float, NDArray[np.float_]] = 1.4
#         ) -> Union[float, NDArray[np.float_]]:
#     """
#     Calculates the final temperature of a gas undergoing an isochoric change
#     due to a pressure step.

#     Args:
#     -----
#     - temperature_initial (float): The initial temperature of the gas in
#     Kelvin.
#     - pressure_initial (float): The initial pressure of the gas in Pascals.
#     - pressure_final (float): The final pressure of the gas in Pascals.
#     - gamma (float): The adiabatic index of the gas, Heat capacity ratio of
#     the gas (Cp/Cv). Default is 1.4.

#     Returns:
#     --------
#     - temperature_final (float): The final temperature of the gas in Kelvin.

#     References:
#     -----------
#     - https://en.wikipedia.org/wiki/Isochoric_process
#     """
#     temperature_final = temperature_initial * (
#         pressure_final / pressure_initial) / gamma
#     return temperature_final
