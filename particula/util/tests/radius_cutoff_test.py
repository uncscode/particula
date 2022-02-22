""" test radius cutoffs
"""

from particula.util.radius_cutoff import cut_rad

def test_cuts():
    """ testing cuts
    """

    assert cut_rad(.9999, .2, 100)[0] <= 100
    assert cut_rad(.9999, .2, 100)[1] >= 100
    assert cut_rad(.9999, .2, 100)[0] <= cut_rad(.9999, .2, 100)[1]
    assert cut_rad(.9999, .2, 100)[0] <= cut_rad(.999, .2, 100)[0]
    assert cut_rad(.999, .2, 100)[1] >= cut_rad(.999, .2, 100)[1]
