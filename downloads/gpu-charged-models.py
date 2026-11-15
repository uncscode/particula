# %% [markdown]
# # Charge-dependent Brownian collisions on CUDA
#
# A charged Brownian kernel is a complete collision model, not a correction
# to add to the ordinary Brownian rate. This example compares the five
# existing Particula models using the same diffusive Knudsen number and
# Coulomb potential. Positive potential means attraction; negative means
# repulsion. Dimensional rates also include Coulomb enhancement factors.
#
# We evaluate the dimensionless fits on CUDA and compare every point with
# the NumPy implementation. Agreement establishes numerical implementation
# parity, not atmospheric accuracy or interchangeability of the fits.
# Ions require separate attachment and recombination validation.

# %%
"""Compare all five charge-dependent collision fits on a real CUDA device."""

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from particula.dynamics.coagulation import charged_dimensionless_kernel as cpu
from particula.gpu.dynamics.coagulation_funcs import charged_dimensionless_wp


@wp.kernel
def evaluate(
    model: wp.int32,
    knudsen: wp.array(dtype=wp.float64),
    potential: wp.array(dtype=wp.float64),
    hard_sphere: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.float64),
):
    """Evaluate concrete device functions without host-side fallback."""
    i = wp.tid()
    result[i] = charged_dimensionless_wp(
        model, knudsen[i], potential[i], hard_sphere[i]
    )


# %% [markdown]
# ## Compare the fitted transition regimes
#
# The panels hold potential fixed and sweep the diffusive Knudsen number.
# All calculations use double precision. The non-attractive branches follow
# each published implementation's convention; the dimensionless curves alone
# must not be interpreted as physical repulsive collision rates.
#
# Dyachkov's nearly neutral expression is evaluated with cancellation-resistant
# differences. Its analytic limit is tested independently in the package.


# %%
def run_example():
    """Verify CUDA parity and plot model differences across three potentials."""
    if not wp.is_cuda_available():
        raise RuntimeError("This example requires CUDA; no CPU fallback")
    models = [
        ("Hard sphere", None),
        ("Dyachkov 2007", cpu.get_coulomb_kernel_dyachkov2007),
        ("Gatti 2008", cpu.get_coulomb_kernel_gatti2008),
        ("Gopalakrishnan 2012", cpu.get_coulomb_kernel_gopalakrishnan2012),
        ("Chahl 2019", cpu.get_coulomb_kernel_chahl2019),
    ]
    kn = np.logspace(-2, 2, 160)
    hard = cpu.get_hard_sphere_kernel(kn)
    device = "cuda:0"
    kn_gpu = wp.array(kn, dtype=wp.float64, device=device)
    hard_gpu = wp.array(hard, dtype=wp.float64, device=device)
    result = wp.empty(kn.size, dtype=wp.float64, device=device)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    maximum_error = 0.0
    for ax, phi in zip(axes, [-1.0, 0.0, 5.0], strict=True):
        potentials = np.full_like(kn, phi)
        phi_gpu = wp.array(potentials, dtype=wp.float64, device=device)
        for model_id, (name, function) in enumerate(models):
            wp.launch(
                evaluate,
                dim=kn.size,
                inputs=[model_id, kn_gpu, phi_gpu, hard_gpu, result],
                device=device,
            )
            observed = result.numpy()
            expected = hard if function is None else function(kn, potentials)
            np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=0)
            maximum_error = max(
                maximum_error, float(np.max(np.abs(observed / expected - 1)))
            )
            ax.loglog(kn, observed, label=name)
        ax.set(
            xlabel="Diffusive Knudsen number",
            ylabel="Dimensionless kernel H",
            title=f"Coulomb potential = {phi:g}",
        )
    axes[-1].legend(fontsize=8)
    fig.suptitle("Charged Brownian models · verified CUDA evaluations")
    print(f"Device: {wp.get_device(device).name}")
    print(f"Maximum relative CPU/CUDA difference: {maximum_error:.3e}")
    print("Parity is numerical evidence, not atmospheric validation.")
    return {"maximum_relative_error": maximum_error}


# %% [markdown]
# ## Use a model in a prepared simulation
#
# Construct `Coagulation(ChargedCoagulationStrategy(
# 'particle_resolved', CoulombGatti2008KernelStrategy()))`, then use its usual
# `prepare`, `run`, `snapshot`, `capture` and checkpoint/resume methods.
# Equal positive active counts are required throughout the run. The model
# selection is frozen at preparation and checked on restart.
#
# The direct interface accepts `CoagulationMechanismConfig(
# mechanisms=('charged_hard_sphere',), charged_model='gatti2008')`.
# The historical mechanism identifier remains for compatibility; the explicit
# model field selects its charged fit. Existing additive requests retain their
# original meaning and should not be used to double-count Brownian collisions.
#
# ## References and scope
#
# - [D'yachkov et al. (2007), Physics of Plasmas 14, 042102](https://doi.org/10.1063/1.2713719):
#   flux matching with a collisionless layer for grain charging.
# - [Gatti and Kortshagen (2008), Physical Review E 78, 046402](https://doi.org/10.1103/PhysRevE.78.046402):
#   a charging model across collisionality regimes.
# - [Gopalakrishnan and Hogan (2012), Physical Review E 85, 026410](https://doi.org/10.1103/PhysRevE.85.026410):
#   the hard-sphere fit in equation (5), charged dimensionless scaling, and
#   comparisons with the earlier charging models. Particula's Gatti expression
#   follows this continuum-consistent formulation rather than independently
#   reimplementing the original plasma model.
# - [Chahl and Gopalakrishnan (2019), Aerosol Science and Technology 53, 933–957](https://doi.org/10.1080/02786826.2019.1614522):
#   high-potential, near-free-molecular Coulombic collision modeling.
#
# These sources distinguish model assumptions and regimes. CPU/CUDA parity
# verifies the port of Particula's formulas, including their existing branch
# conventions; it does not validate all five fits for atmospheric small ions.

# %%
if __name__ == "__main__":
    run_example()
