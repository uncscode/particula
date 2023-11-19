"""Calculates the Gibbs free energy of mixing for a binary solution."""

from particula.activity.machine_limit import safe_log


def gibbs_free_engery(
    org_mole_fraction,
    gibbs_mix,
):
    """
    Calculate the gibbs free energy of the mixture. Ideal and non-ideal.

    Args:
    org_mole_fraction (np.array): A numpy array of organic mole fractions.
    gibbs_mix (np.array): A numpy array of gibbs free energy of mixing.

    Returns:
    gibbs_ideal (np.array): The ideal gibbs free energy of mixing.
    gibbs_real (np.array): The real gibbs free energy of mixing.
    """

    gibbs_ideal = (1 - org_mole_fraction) \
        * safe_log(1 - org_mole_fraction) \
        + org_mole_fraction \
        * safe_log(org_mole_fraction)
    gibbs_real = gibbs_ideal + gibbs_mix
    return gibbs_ideal, gibbs_real
