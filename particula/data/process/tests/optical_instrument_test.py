"""
Test for optical instrument module.
"""

from particula.data.process.optical_instrument import (
    CapsInstrumentKeywordBuilder,
)


def test_set_keyword():
    """Test setting a single keyword."""
    builder = CapsInstrumentKeywordBuilder()
    builder.set_keyword("caps_extinction_dry", "CAPS_extinction_dry_[1/Mm]")
    assert (
        builder.keyword_dict["caps_extinction_dry"]
        == "CAPS_extinction_dry_[1/Mm]"
    )


def test_set_keywords():
    """Test setting multiple keywords."""
    builder = CapsInstrumentKeywordBuilder()
    builder.set_keywords(
        caps_extinction_dry="CAPS_extinction_dry_[1/Mm]",
        caps_extinction_wet="CAPS_extinction_wet_[1/Mm]",
        caps_scattering_dry="CAPS_scattering_dry_[1/Mm]",
        caps_scattering_wet="CAPS_scattering_wet_[1/Mm]",
    )
    assert builder.keyword_dict["caps_extinction_dry"] == (
        "CAPS_extinction_dry_[1/Mm]")
    assert builder.keyword_dict["caps_extinction_wet"] == (
        "CAPS_extinction_wet_[1/Mm]")
    assert builder.keyword_dict["caps_scattering_dry"] == (
        "CAPS_scattering_dry_[1/Mm]")
    assert builder.keyword_dict["caps_scattering_wet"] == (
        "CAPS_scattering_wet_[1/Mm]")
    assert builder.keyword_dict["caps_relative_humidity_dry"] is None


def test_partial_build():
    """Test building some keywords."""
    partial_build = (
        CapsInstrumentKeywordBuilder()
        .set_keyword("caps_extinction_dry", "CAPS_extinction_dry_[1/Mm]")
        .set_keyword("caps_extinction_wet", "CAPS_extinction_wet_[1/Mm]")
        .set_keyword("caps_scattering_dry", "CAPS_scattering_dry_[1/Mm]")
        .set_keyword("caps_scattering_wet", "CAPS_scattering_wet_[1/Mm]")
        )

    # check error on partial build
    try:
        partial_build.build()
    except ValueError as e:
        assert str(e) == (
            "The following keywords have not been set: "
            "caps_relative_humidity_dry, caps_relative_humidity_wet, "
            "sizer_relative_humidity, refractive_index_dry, "
            "water_refractive_index"
        )


def test_full_build():
    """Test building all keywords."""
    keywords = (
        CapsInstrumentKeywordBuilder()
        .set_keyword("caps_extinction_dry", "CAPS_extinction_dry_[1/Mm]")
        .set_keyword("caps_extinction_wet", "CAPS_extinction_wet_[1/Mm]")
        .set_keyword("caps_scattering_dry", "CAPS_scattering_dry_[1/Mm]")
        .set_keyword("caps_scattering_wet", "CAPS_scattering_wet_[1/Mm]")
        .set_keyword("caps_relative_humidity_dry",
                     "CAPS_relative_humidity_dry")
        .set_keyword("caps_relative_humidity_wet",
                     "CAPS_relative_humidity_wet")
        .set_keyword("sizer_relative_humidity", "RH_percent")
        .set_keyword("refractive_index_dry", 1.5)
        .set_keyword("water_refractive_index", 1.33)
        .set_keyword("wavelength", 450)
        .set_keyword("discretize_kappa_fit", True)
        .set_keyword("discretize_truncation", True)
        .set_keyword("fit_kappa", True)
        .set_keyword("calculate_truncation", True)
        .set_keyword("calibration_dry", 1.0)
        .set_keyword("calibration_wet", 1.0)
    ).build()

    assert keywords["caps_extinction_dry"] == "CAPS_extinction_dry_[1/Mm]"
    assert keywords["caps_extinction_wet"] == "CAPS_extinction_wet_[1/Mm]"
    assert keywords["caps_scattering_dry"] == "CAPS_scattering_dry_[1/Mm]"
    assert keywords["caps_scattering_wet"] == "CAPS_scattering_wet_[1/Mm]"
    assert keywords["caps_relative_humidity_dry"] == (
        "CAPS_relative_humidity_dry")
    assert keywords["caps_relative_humidity_wet"] == (
        "CAPS_relative_humidity_wet")
    assert keywords["sizer_relative_humidity"] == "RH_percent"
    assert keywords["refractive_index_dry"] == 1.5
    assert keywords["water_refractive_index"] == 1.33
    assert keywords["wavelength"] == 450.0
    assert keywords["discretize_kappa_fit"] is True
    assert keywords["discretize_truncation"] is True
    assert keywords["fit_kappa"] is True
    assert keywords["calculate_truncation"] is True
    assert keywords["calibration_dry"] == 1.0
    assert keywords["calibration_wet"] == 1.0
