"""Convert between different size distribution formats.

This module defines conversion strategies as classes that implement
the ConversionStrategy interface. The SizerConverter composes a
ConversionStrategy instance to flexibly convert distribution data among
scales.

Examples:
    ``` py title="Example usage"
    import numpy as np
    from particula.util.size_distribution_convert import (
        get_conversion_strategy, SizerConverter
    )

    diameters = np.array([1e-7, 1e-6, 1e-5])
    concentration = np.array([1e6, 1e5, 1e4])

    strategy = get_conversion_strategy("dn/dlogdp", "PMF")
    converter = SizerConverter(strategy)
    converted_conc = converter.convert(diameters, concentration)
    print(converted_conc)
    # Output: [...]
    ```

References:
    - W. C. Hinds, "Aerosol Technology," 2nd ed., Wiley-Interscience, 1999.

To be moved to particle.properties. -kyle
"""

# pylint: disable=too-few-public-methods

import numpy as np


class ConversionStrategy:
    """Conversion strategy interface for particle size distribution data.

    Methods:
        - convert: Convert distribution data between input and output scales.
    Defines an interface for conversion strategies between particle size
    distribution formats.

    All subclasses must implement the convert method to perform the actual
    conversion logic.

    Examples:
        ``` py title="Subclass Example"
        class CustomStrategy(ConversionStrategy):
            def convert(self, diameters, concentration, inverse=False):
                # Custom conversion logic here
                return concentration
        ```
    """

    def convert(
        self,
        diameters: np.ndarray,
        concentration: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Convert distribution data from one scale to another.

        Arguments:
            - diameters : Array of particle diameters.
            - concentration : The distribution data corresponding to these
                diameters.
            - inverse : If True, reverse the direction of the conversion.

        Returns:
            - np.ndarray of converted distribution data.

        Raises:
            - NotImplementedError : If not overridden by a subclass.
        """
        raise NotImplementedError(
            "This method should be overridden by subclasses."
        )


class SameScaleConversionStrategy(ConversionStrategy):
    """Conversion strategy that returns the input concentration unchanged.

    No conversion is performed because the input and output scales are the
    same.

    Examples:
        ```py title="Example Usage"
        strategy = SameScaleConversionStrategy()
        result = strategy.convert(diameters, concentration)
        # result is identical to concentration
        ```
    """

    def convert(
        self,
        diameters: np.ndarray,
        concentration: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Return the concentration unchanged, since no conversion is needed.

        Arguments:
            - diameters : Array of particle diameters (unused).
            - concentration : The original distribution data.
            - inverse : Flag indicating direction (unused).

        Returns:
            - np.ndarray identical to the input concentration.
        """
        return concentration


class DNdlogDPtoPMFConversionStrategy(ConversionStrategy):
    """Conversion strategy for converting between dn/dlogdp and PMF formats.

    Examples:
        ``` py title="Example Usage"
        strategy = DNdlogDPtoPMFConversionStrategy()
        result = strategy.convert(diameters, dn_dlogdp_conc)
        # result is now in PMF format
        ```
    """

    def convert(
        self,
        diameters: np.ndarray,
        concentration: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Perform the conversion between dn/dlogdp and PMF formats.

        Arguments:
            - diameters : Array of particle diameters.
            - concentration : Distribution data in dn/dlogdp or PMF format.
            - inverse : If True, convert from PMF to dn/dlogdp; otherwise the
                opposite.

        Returns:
            - np.ndarray of the distribution in the target format.
        """
        return get_distribution_in_dn(diameters, concentration, inverse=inverse)


class PMFtoPDFConversionStrategy(ConversionStrategy):
    """Conversion strategy for converting between PMF and PDF formats.

    Examples:
        ``` py title="Example Usage"
        strategy = PMFtoPDFConversionStrategy()
        result_pdf = strategy.convert(diameters, PMF_data, inverse=False)
        # result_pdf is now in PDF format
        ```
    """

    def convert(
        self,
        diameters: np.ndarray,
        concentration: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Perform the conversion between PMF and PDF formats.

        Arguments:
            - diameters : Array of particle diameters.
            - concentration : Distribution data in PMF or PDF format.
            - inverse : If True, convert from PDF to PMF; otherwise from PMF
                to PDF.

        Returns:
            - np.ndarray of the distribution in the target format.
        """
        return get_pdf_distribution_in_pmf(
            diameters, concentration, to_pdf=not inverse
        )


class DNdlogDPtoPDFConversionStrategy(ConversionStrategy):
    """Conversion strategy for converting between dn/dlogdp and PDF formats.

    This strategy first converts dn/dlogdp to PMF, then to PDF, or the
    reverse if inverse is True.

    Examples:
        ``` py title="Example Usage"
        strategy = DNdlogDPtoPDFConversionStrategy()
        result_pdf = strategy.convert(diameters, dn_dlogdp_data)
        # result_pdf is now in PDF format
        ```
    """

    def convert(
        self,
        diameters: np.ndarray,
        concentration: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Convert between dn/dlogdp and PDF formats.

        This method first converts dn/dlogdp to PMF, then to PDF, or the
        reverse if inverse is True.

        Arguments:
            - diameters : Array of particle diameters.
            - concentration : Distribution data in dn/dlogdp or PDF format.
            - inverse : If True, convert from PDF to dn/dlogdp; otherwise the
                opposite.

        Returns:
            - np.ndarray of the distribution in the target format.
        """
        if inverse:
            concentration_pmf = get_pdf_distribution_in_pmf(
                diameters, concentration, to_pdf=False
            )
            return get_distribution_in_dn(
                diameters, concentration_pmf, inverse=True
            )
        # if not inverse, then dn/dlogdp to PDF
        concentration_pmf = get_distribution_in_dn(
            diameters, concentration, inverse=False
        )
        return get_pdf_distribution_in_pmf(
            diameters, concentration_pmf, to_pdf=True
        )


class SizerConverter:
    """A converter that composes a ConversionStrategy.

    This class allows converting particle size distribution data between
    different formats using a specified conversion strategy.
    It provides a flexible interface to convert data without needing to
    know the details of the conversion logic.

    Examples:
        ``` py title="Example Usage"
        diameters = [1e-7, 1e-6, 1e-5]
        concentration = [1e6, 1e5, 1e4]

        strategy = DNdlogDPtoPMFConversionStrategy()
        converter = SizerConverter(strategy)
        new_conc = converter.convert(diameters, concentration)
        ```
    """

    def __init__(self, strategy: ConversionStrategy):
        """Initializes the converter with a conversion strategy.

        Args:
            strategy (ConversionStrategy): The strategy to use for conversion.
        """
        self.strategy = strategy

    def convert(
        self,
        diameters: np.ndarray,
        concentration: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray:
        """Convert the particle size distribution data.

        Arguments:
            - diameters : Array of particle diameters.
            - concentration : Distribution data.
            - inverse : If True, reverse the conversion direction
                (if supported).

        Returns:
            - np.ndarray of the converted distribution.
        """
        return self.strategy.convert(diameters, concentration, inverse=inverse)


def get_distribution_conversion_strategy(
    input_scale: str, output_scale: str
) -> ConversionStrategy:
    """Factory function to obtain a conversion strategy.

    Arguments:
        - input_scale : Scale of the input distribution, e.g.
            'dn/dlogdp' or 'pmf'.
        - output_scale : Desired scale of the output distribution, e.g.
            'pmf' or 'pdf'.

    Returns:
        - A ConversionStrategy object supporting the requested conversion.

    Raises:
        - ValueError : If scales are invalid or unsupported.

    Examples:
        ``` py title="Example Usage"
        strategy = get_distribution_conversion_strategy('dn/dlogdp', 'pdf')
        converter = SizerConverter(strategy)
        converted_data = converter.convert(diameters, concentration)
        ```
    """
    # force lower case scales
    input_scale = input_scale.lower()
    output_scale = output_scale.lower()

    if input_scale == output_scale:  # early return for same scales
        return SameScaleConversionStrategy()

    # Validate input and output scales
    valid_input_scales = ["dn/dlogdp", "pmf"]
    valid_output_scales = ["pmf", "pdf"]
    if input_scale not in valid_input_scales:
        raise ValueError(
            f"input_scale '{input_scale}' is not supported."
            + f"Must be one of {valid_input_scales}."
            "Inverse flag can invert the input-output."
        )

    if output_scale not in valid_output_scales:
        raise ValueError(
            f"output_scale '{output_scale}' is not supported."
            + f"Must be one of {valid_output_scales}."
            "Inverse flag can invert the input-output."
        )

    # Determine and return the appropriate conversion strategy
    if input_scale == "dn/dlogdp" and output_scale == "pmf":
        return DNdlogDPtoPMFConversionStrategy()
    if input_scale == "pmf" and output_scale == "pdf":
        return PMFtoPDFConversionStrategy()
    if input_scale == "dn/dlogdp" and output_scale == "pdf":
        return DNdlogDPtoPDFConversionStrategy()
    raise ValueError(
        f"Unsupported conversion from {input_scale} to {output_scale}"
    )


def get_distribution_in_dn(
    diameter: np.ndarray, dn_dlogdp: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """Convert the sizer data between dn/dlogdp and d_num formats.

    If inverse=False, this function applies:
    - d_num = dn_dlogdp × (log10(upper / lower))
        - The bin width is determined by upper and lower diameter limits,
          with log10 scaling.

    If inverse=True, it reverts:
    - dn/dlogdp = d_num / (log10(upper / lower))

    Arguments:
        - diameter : Array of particle diameters.
        - dn_dlogdp : Array representing either dn/dlogdp or d_num.
        - inverse : If True, converts from d_num to dn/dlogdp; otherwise the
            opposite.

    Returns:
        - A np.ndarray of the converted distribution.

    Examples:
        ```py
        import numpy as np
        from particula.util.size_distribution_convert import convert_sizer_dn

        diam = np.array([1e-7, 2e-7, 4e-7])
        dn_logdp = np.array([1e6, 1e5, 1e4])
        result = convert_sizer_dn(diam, dn_logdp, inverse=False)
        print(result)
        # Output: d_num format for each diameter bin
        ```

    References:
        - "dN/dlogD_p and dN/dD_p," TSI Application Note PR-001, 2010.
            [link](
            https://tsi.com/getmedia/1621329b-f410-4dce-992b-e21e1584481a/
            PR-001-RevA_Aerosol-Statistics-AppNote?ext=.pd)
    """
    # Validate inputs are non-empty arrays
    if len(diameter) == 0 or len(dn_dlogdp) == 0:
        raise TypeError("Both diameter and dn_dlogdp must be numpy arrays.")

    # Compute the bin widths
    delta = np.zeros_like(diameter)
    delta[:-1] = np.diff(diameter)
    delta[-1] = delta[-2] ** 2 / delta[-3]

    # Compute the lower and upper bin edges
    lower = diameter - delta / 2
    upper = diameter + delta / 2

    if inverse:
        # Convert from dn to dn/dlogdp
        return dn_dlogdp / np.log10(upper / lower)

    return dn_dlogdp * np.log10(upper / lower)


def get_pdf_distribution_in_pmf(
    x_array: np.ndarray, distribution: np.ndarray, to_pdf: bool = True
) -> np.ndarray:
    """Convert from PMF to PDF or vice versa.

    The conversion uses:
    - y_pdf = y_PMF / Δx
    - y_PMF = y_pdf * Δx
      - Δx is the bin width, determined by consecutive differences in x_array.

    Arguments:
        - x_array : An array of diameters/radii for the distribution bins.
        - distribution : The original distribution data (PMF or PDF).
        - to_pdf : If True, convert from PMF to PDF; if False, from PDF to PMF.

    Returns:
        - A np.ndarray of the converted distribution data.

    Examples:
        ```py
        import numpy as np
        import particula as par
        x_vals = np.array([1.0, 2.0, 3.0])
        PMF = np.array([10.0, 5.0, 2.5])
        pdf = par.get_pdf_distribution_in_pmf(x_vals, PMF, to_pdf=True)
        print(pdf)
        # Output: [10.  5.  2.5] / [1.0, 1.0, ...] = ...
        ```

    References:
        - Detailed bin width discussion in: TSI Application Note
          "Aerosol Statistics and Densities."
    """
    # Calculate the differences between consecutive x_array values for bin
    # widths.
    delta_x_array = np.empty_like(x_array)
    delta_x_array[:-1] = np.diff(x_array)
    # For the last bin, extrapolate the width assuming constant growth
    # rate from the last two bins.
    delta_x_array[-1] = delta_x_array[-2] ** 2 / delta_x_array[-3]

    # Converting PMF to PDF by dividing the PMF values by the bin widths. or
    # Converting PDF to PMF by multiplying the PDF values by the bin widths.
    return (
        distribution / delta_x_array if to_pdf else distribution * delta_x_array
    )
