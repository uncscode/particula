"""
Blending weights for the BAT model.
"""


def test_bat_blending_weights():
    """test for bat_blending_weights function."""
    molar_mass_ratio = 18.016 / 250
    oxygen2carbon = 0.3

    weights = binary_activity.bat_blending_weights(
        molar_mass_ratio=molar_mass_ratio, oxygen2carbon=oxygen2carbon
    )
    assert np.all(weights >= 0)
    assert np.all(weights <= 1)
