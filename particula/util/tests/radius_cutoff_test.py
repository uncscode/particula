""" Testing the radius cutoff utility
"""

from particula.util.radius_cutoff import cut_rad


def test_cuts():
    """ testing cuts:
            * test if starting radius is lower than mode
            * test if ending radius is higher than mod
            * test if lower radius is smaller than end radius
            * test if lower radius is smaller when cutoff is smaller
            * test if ending radius is larger when cutoff is larger
            * test if ending radius is larger when gsigma is higher
    """

    assert cut_rad(cutoff=.9999, gsigma=1.25, mode=1e-7)[0] <= 1e-7

    assert cut_rad(cutoff=.9999, gsigma=1.25, mode=1e-7)[1] >= 1e-7

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=1e-7)[0]
        <=
        cut_rad(cutoff=.9999, gsigma=1.25, mode=1e-7)[1]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=1e-7)[0]
        <=
        cut_rad(cutoff=.9990, gsigma=1.25, mode=1e-7)[0]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=1e-7)[1]
        >=
        cut_rad(cutoff=.9990, gsigma=1.25, mode=1e-7)[1]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=1e-7)[1]
        <=
        cut_rad(cutoff=.9999, gsigma=1.35, mode=1e-7)[1]
    )


def test_multi_cuts():
    """ test case for different modes
    """

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[0]
        <=
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[1]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[0]
        <=
        cut_rad(cutoff=.9990, gsigma=1.25, mode=[1e-7, 1e-8])[0]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[1]
        >=
        cut_rad(cutoff=.9990, gsigma=1.25, mode=[1e-7, 1e-8])[1]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[1]
        <=
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[1]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[1]
        ==
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7])[1]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[1]
        >=
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-8])[1]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[0]
        ==
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-8])[0]
    )

    assert (
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7, 1e-8])[0]
        <=
        cut_rad(cutoff=.9999, gsigma=1.25, mode=[1e-7])[0]
    )
