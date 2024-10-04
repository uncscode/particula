"""Converts between different size distribution formats
written with composition in mind and using a factory pattern.

Converter does not inherit from different classes for different conversion
behaviors. Instead, it composes its behavior by holding a ConversionStrategy
object, which can be any strategy conforming to the ConversionStrategy
interface. This is an example of "composition over inheritance",
where behavior is composed at runtime through objects rather than fixed at
compile-time through class hierarchies.

This design allows for flexible swapping of conversion strategies without
needing to modify the Converter class, adhering more closely to the principle
of composition over inheritance and enhancing the code's flexibility and
extensibility.
"""
# pylint: disable=too-few-public-methods


import numpy as np
from particula.util import convert


class ConversionStrategy:
    """Defines an interface for conversion strategies between particle size
    distribution formats.

    Subclasses must implement the convert method to perform specific
    conversion logic.
    """

    def convert(self,
                diameters: np.ndarray,
                concentration: np.ndarray,
                inverse: bool = False) -> np.ndarray:
        """Converter method common interface, for subclasses.

        Args:
            diameters (np.ndarray): The particle diameters.
            concentration (np.ndarray): The concentration values.
            inverse (bool): Flag to perform the inverse conversion.

        Returns:
            np.ndarray: The concentration values converted.

        Raises:
            NotImplementedError: If the subclass does not implement this.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses.")


class SameScaleConversionStrategy(ConversionStrategy):
    """Implements conversion between the same scales, which is a no-op."""

    def convert(self, diameters: np.ndarray, concentration: np.ndarray,
                inverse: bool = False) -> np.ndarray:
        # No conversion needed, return the input concentration
        return concentration


class DNdlogDPtoPMSConversionStrategy(ConversionStrategy):
    """Implements conversion between dn/dlogdp and PMS formats using the
    convert_sizer_dn method."""

    def convert(self, diameters: np.ndarray, concentration: np.ndarray,
                inverse: bool = False) -> np.ndarray:
        # Call the conversion utility function, handling direct and inverse
        # conversions
        return convert.convert_sizer_dn(
            diameters, concentration, inverse=inverse)


class PMStoPDFConversionStrategy(ConversionStrategy):
    """Implements conversion between PMS and PDF formats."""

    def convert(self, diameters: np.ndarray, concentration: np.ndarray,
                inverse: bool = False) -> np.ndarray:
        # Converts to PDF or back to PMS based on the inverse flag
        return convert.distribution_convert_pdf_pms(
            diameters, concentration, to_pdf=not inverse)


class DNdlogDPtoPDFConversionStrategy(ConversionStrategy):
    """Implements conversion between dn/dlogdp and PDF formats through an
    intermediate PMS format."""

    def convert(self, diameters: np.ndarray, concentration: np.ndarray,
                inverse: bool = False) -> np.ndarray:
        # if the inverse, then to PDF to dn/dlogdp
        if inverse:
            concentration_pms = convert.distribution_convert_pdf_pms(
                diameters, concentration, to_pdf=False)
            return convert.convert_sizer_dn(
                diameters, concentration_pms, inverse=True)
        # if not inverse, then dn/dlogdp to PDF
        concentration_pms = convert.convert_sizer_dn(
            diameters, concentration, inverse=False)
        return convert.distribution_convert_pdf_pms(
            diameters, concentration_pms, to_pdf=True)


class SizerConverter:
    """A converter that uses a specified ConversionStrategy to convert
    particle size distribution data between different formats."""

    def __init__(self, strategy: ConversionStrategy):
        """Initializes the converter with a conversion strategy.

        Args:
            strategy (ConversionStrategy): The strategy to use for conversion.
        """
        self.strategy = strategy

    def convert(self, diameters: np.ndarray, concentration: np.ndarray,
                inverse: bool = False) -> np.ndarray:
        """Converts particle size distribution data using the specified
        strategy.

        Args:
            diameters (np.ndarray): The particle diameters.
            concentration (np.ndarray): The concentration values.
            inverse (bool): Flag to perform the inverse conversion.

        Returns:
            np.ndarray: The converted concentration values.
        """
        return self.strategy.convert(diameters, concentration, inverse=inverse)


def get_conversion_strategy(input_scale: str,
                            output_scale: str) -> ConversionStrategy:
    """Factory function to create and return an appropriate conversion
    strategy based on input and output scales. Use the inverse flag in the
    converter to invert the directions of the input and output scales.

    Args:
        input_scale: The scale of the input concentration values.
            Either 'dn/dlogdp' or 'pms'.
        output_scale: The desired scale of the output concentration
            values. Either 'pms' or 'pdf'. Use inverse flag to invert the input
            and output scales.

    Returns:
        ConversionStrategy: A strategy object capable of converting between
            the specified scales.

    Raises:
        ValueError: If the input_scale or output_scale is not supported, or
            if the specified conversion is unsupported.

    Example:
        ``` py title="Convert dn/dlogdp to PMS"
        strategy = get_conversion_strategy('dn/dlogdp', 'pms')
        converter = Converter(strategy)
        converted_concentration = converter.convert(
            diameters, concentration, inverse=False)
        ```
    """
    # force lower case scales
    input_scale = input_scale.lower()
    output_scale = output_scale.lower()

    if input_scale == output_scale:  # early return for same scales
        return SameScaleConversionStrategy()

    # Validate input and output scales
    valid_input_scales = ['dn/dlogdp', 'pms']
    valid_output_scales = ['pms', 'pdf']
    if input_scale not in valid_input_scales:
        raise ValueError(
            f"input_scale '{input_scale}' is not supported."
            + f"Must be one of {valid_input_scales}."
            "Inverse flag can invert the input-output.")

    if output_scale not in valid_output_scales:
        raise ValueError(
            f"output_scale '{output_scale}' is not supported."
            + f"Must be one of {valid_output_scales}."
            "Inverse flag can invert the input-output.")

    # Determine and return the appropriate conversion strategy
    if input_scale == 'dn/dlogdp' and output_scale == 'pms':
        return DNdlogDPtoPMSConversionStrategy()
    if input_scale == 'pms' and output_scale == 'pdf':
        return PMStoPDFConversionStrategy()
    if input_scale == 'dn/dlogdp' and output_scale == 'pdf':
        return DNdlogDPtoPDFConversionStrategy()
    raise ValueError(
        f"Unsupported conversion from {input_scale} to {output_scale}")
