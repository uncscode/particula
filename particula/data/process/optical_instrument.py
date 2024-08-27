"""Functions for processing optical data."""

from typing import Optional, Union
import logging
import numpy as np

from particula.data.stream import Stream
from particula.data.process import scattering_truncation, kappa_via_extinction

logger = logging.getLogger("particula")


class CapsInstrumentKeywordBuilder:
    """Builder class for CAPS Instrument Keywords dictionary."""

    _default_keywords = {
        "caps_extinction_dry": None,
        "caps_extinction_wet": None,
        "caps_scattering_dry": None,
        "caps_scattering_wet": None,
        "caps_relative_humidity_dry": None,
        "caps_relative_humidity_wet": None,
        "sizer_relative_humidity": None,
        "refractive_index_dry": None,
        "water_refractive_index": None,
        "wavelength": 450,
        "discretize_kappa_fit": True,
        "discretize_truncation": True,
        "fit_kappa": True,
        "calculate_truncation": True,
        "calibration_dry": 1.0,
        "calibration_wet": 1.0,
    }

    def __init__(self):
        self.keyword_dict = self._default_keywords.copy()

    def set_keyword(
        self, keyword: str, value: Optional[Union[str, float, int, bool]]
    ):
        """Set the keyword parameter for the activity calculation.

        Args:
            keyword: The keyword to set.
            value: The value to set the keyword to.

        Raises:
            ValueError: If the keyword is not recognized or the value type
                is incorrect.
        """
        if keyword not in self.keyword_dict:
            value_error = f"Unrecognized keyword: {keyword}"
            logger.error(value_error)
            raise ValueError(value_error)

        # Optionally, add type checking or value validation here
        if not isinstance(value, (str, float, bool, int)):
            type_message = (
                f"Invalid value type for keyword '{keyword}': {type(value)}"
            )
            logger.error(type_message)
            raise TypeError(type_message)

        self.keyword_dict[keyword] = value  # type: ignore
        return self

    def set_keywords(self, **kwargs: Union[str, float, int, bool]):
        """Set multiple keywords at once.

        Args:
            kwargs: The keywords and their values to set.
        """
        for keyword, value in kwargs.items():
            self.set_keyword(keyword, value)
        return self  # Enable method chaining

    def pre_build_check(self):
        """Check that all required parameters have been set.

        Raises:
            ValueError: If any required keyword has not been set.
        """
        unset_keywords = [k for k, v in self.keyword_dict.items() if v is None]
        if unset_keywords:
            value_message = (
                "The following keywords have not been set: "
                f"{', '.join(unset_keywords)}"
            )
            logger.error(value_message)
            raise ValueError(value_message)

    def build(self) -> dict[str, Union[str, float, int, bool]]:
        """Validate and return the keywords dictionary.

        Returns:
            dict: The validated keywords dictionary.
        """
        self.pre_build_check()
        return self.keyword_dict  # type: ignore


def caps_processing(
        stream_size_distribution: Stream,
        stream_sizer_properties: Stream,
        stream_caps: Stream,
        keywords: dict[str, Union[str, float, int, bool]]
):
    """
    Function to process the CAPS data, and smps for kappa fitting, and then add
    it to the datalake. Also applies truncation corrections to the CAPS data.

    Args:

    Returns:
    """
    # calc kappa and add to datalake
    logger.info("Processing CAPS data")

    if keywords['fit_kappa']:
        kappa_matrix = \
            kappa_via_extinction.kappa_from_extinction_looped(
                extinction_dry=stream_caps[keywords['caps_extinction_dry']],
                extinction_wet=stream_caps[keywords['caps_extinction_wet']],
                number_per_cm3=stream_size_distribution.data,
                diameter=stream_size_distribution.header_float,
                water_activity_sizer=stream_sizer_properties[
                    keywords['sizer_relative_humidity']]/100,
                water_activity_sample_dry=stream_caps[
                    keywords['caps_relative_humidity_dry']]/100,
                water_activity_sample_wet=stream_caps[
                    keywords['caps_relative_humidity_wet']]/100,
                refractive_index_dry=keywords['refractive_index_dry'],
                water_refractive_index=keywords['water_refractive_index'],
                wavelength=keywords['wavelength'],
                discretize=keywords['discretize_kappa_fit']
            )

    else:
        kappa_fixed = keywords.get('kappa_fixed', 0.1)
        kappa_matrix = np.ones(
            (len(stream_caps.time), 3), dtype=np.float64) * kappa_fixed

    # add kappa_matrix to datalake
    stream_caps['kappa_fit'] = kappa_matrix[:, 0]
    stream_caps['kappa_fit_lower'] = kappa_matrix[:, 1]
    stream_caps['kappa_fit_upper'] = kappa_matrix[:, 2]

    # check if raw in dict
    index_dic = stream_caps.header_dict
    if "raw_" + keywords["caps_scattering_dry"] in index_dic:
        pass
    else:
        # copy and save raw data
        stream_caps["raw_" + keywords["caps_scattering_dry"]] = stream_caps[
            keywords["caps_scattering_dry"]
        ]
        stream_caps["raw_" + keywords["caps_scattering_dry"]] = stream_caps[
            keywords["caps_scattering_wet"]
        ]

    # reset the stream data to the raw data
    stream_caps[keywords["caps_scattering_dry"]] = stream_caps[
        "raw_" + keywords["caps_scattering_dry"]
    ]
    stream_caps[keywords["caps_scattering_wet"]] = stream_caps[
        "raw_" + keywords["caps_scattering_wet"]
    ]

    # apply calibration factors
    stream_caps[keywords["caps_scattering_dry"]] *= keywords["calibration_dry"]
    stream_caps[keywords["caps_scattering_wet"]] *= keywords["calibration_wet"]

    # calc truncation corrections and add to datalake
    logger.info("CAPS truncation corrections")
    if keywords['calculate_truncation']:

        truncation_dry = \
            scattering_truncation.correction_for_humidified_looped(
                kappa=kappa_matrix[:, 0],
                number_per_cm3=stream_size_distribution.data,
                diameter=stream_size_distribution.header_float,
                water_activity_sizer=stream_sizer_properties[
                    keywords['sizer_relative_humidity']]/100,
                water_activity_sample=stream_caps[
                    keywords['caps_relative_humidity_dry']]/100,
                refractive_index_dry=keywords['refractive_index_dry'],
                water_refractive_index=keywords['water_refractive_index'],
                wavelength=keywords['wavelength'],
                discretize=keywords['discretize_truncation']
            )

        truncation_wet = \
            scattering_truncation.correction_for_humidified_looped(
                kappa=kappa_matrix[:, 0],
                number_per_cm3=stream_size_distribution.data,
                diameter=stream_size_distribution.header_float,
                water_activity_sizer=stream_sizer_properties[
                    keywords['sizer_relative_humidity']]/100,
                water_activity_sample=stream_caps[
                    keywords['caps_relative_humidity_wet']]/100,
                refractive_index_dry=keywords['refractive_index_dry'],
                water_refractive_index=keywords['water_refractive_index'],
                wavelength=keywords['wavelength'],
                discretize=keywords['discretize_truncation']
            )

        # apply truncation corrections
        stream_caps[keywords['caps_scattering_dry']] *= truncation_dry
        stream_caps[keywords['caps_scattering_wet']] *= truncation_wet
    else:
        truncation_dry = np.ones_like(stream_caps.time, dtype=np.float64)
        truncation_wet = np.ones_like(stream_caps.time, dtype=np.float64)

    # add truncation corrections to stream
    stream_caps['truncation_dry'] = truncation_dry
    stream_caps['truncation_wet'] = truncation_wet

    return stream_caps


def albedo_from_ext_scat(
    stream: Stream,
    extinction_key: str,
    scattering_key: str,
    new_absorption_key: str,
    new_albedo_key: str,
) -> Stream:
    """
    Calculates the albedo from the extinction and scattering data in Stream.

    The function computes the absorption as the difference between extinction
    and scattering, and the single-scattering albedo as the ratio of
    scattering to extinction. If the extinction values are zero or negative,
    the albedo is set to np.nan. The user can choose to filter out
    zero or negative extinction values before calculation.

    Args:
        stream (Stream): The datastream containing CAPS data.
        extinction_key (str): The key for the extinction data in the stream.
        scattering_key (str): The key for the scattering data in the stream.
        new_absorption_key (str): The key where the calculated absorption will
            be stored.
        new_albedo_key (str): The key where the calculated albedo will
            be stored.

    Returns:
        Stream: The updated datastream with the new absorption and albedo
            values.

    Raises:
        KeyError: If the provided extinction or scattering keys are not found
            in the stream.
    """
    if extinction_key not in stream.header:
        key_message = (
            f"Extinction key '{extinction_key}' not found in the stream."
        )
        logger.error(key_message)
        raise KeyError(key_message)

    if scattering_key not in stream.header:
        key_message = (
            f"Scattering key '{scattering_key}' not found in the stream."
        )
        logger.error(key_message)
        raise KeyError(key_message)

    # Calculate absorption
    stream[new_absorption_key] = (
        stream[extinction_key] - stream[scattering_key]
    )

    # Initialize albedo with NaN
    albedo = np.full_like(stream[extinction_key], np.nan)

    # Calculate albedo
    select = stream[extinction_key] > 0
    albedo = stream[scattering_key][select] / stream[extinction_key][select]

    # Store albedo in the stream
    stream[new_albedo_key] = albedo

    return stream
