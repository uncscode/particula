""" radius cutoff
"""


from scipy.stats import lognorm


def cut_rad(cut, sig, scl):
    """ cutoffs
    """

    (rad_start, rad_end) = lognorm.interval(
            alpha=cut,
            s=sig,
            scale=scl,
        )

    return (rad_start, rad_end)
