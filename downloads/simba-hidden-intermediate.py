# %% [markdown]
# # Can a delayed product reveal a hidden intermediate?
#
# Imagine injecting 1 µmol/m³ of A into a closed gas chamber. We can
# measure P at several times, but the instrument cannot see an intermediate
# I. A direct conversion A → P and a two-step conversion A → I → P both
# conserve the injected mass. Their predictions for the first seconds
# differ: the intermediate must accumulate before it can supply product.
# Here we generate synthetic measurements with an analytical solution and
# fit both candidate networks. We then repeat the comparison with an
# instrument that measures only total mass.
#
# ## Choose the reactions the instrument must distinguish
#
# Each row below gives the molecular changes in the order A, I, P. Each
# allowed conversion preserves one unit of the same conserved material.
# We permit up to two steps and require the net conversion of A to P.
# The returned list includes the direct step and the hidden chain.
# Exhausting this small search allows us to compare every feasible network
# within these limits.

# %%
import matplotlib.pyplot as plt
import numpy as np
from particula.dynamics.chemistry.discovery import enumerate_networks
from particula.dynamics.chemistry.inference import (
    ExperimentLoss,
    fit_experiments,
)
from scipy.linalg import expm

TIMES = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
SIGMA = 0.015  # concentration scale (µmol/m³) used to weight residuals


def candidates():
    """Enumerate the networks from the common precursor to product balance."""
    return enumerate_networks(
        steps=[(-1, 0, 1), (-1, 1, 0), (0, -1, 1)],
        initial=[True, False, False],
        products=[2],
        max_steps=2,
        max_trials=10,
        max_candidates=4,
        net=(-1, 0, 1),
    )


# %% [markdown]
# ## Predict product and total concentrations
#
# For independent synthetic data we prescribe first-order rates of 0.5
# and 1.3 s⁻¹. In a sequential chain, A decays exponentially, I is the
# difference of two exponentials and P contains the remaining inventory.
# The fitting model uses the enumerated reaction matrix directly. Its
# matrix exponential integrates the first-order system for any trial
# rates. All species have the same molar mass, so the sum of their
# concentrations is proportional to the chamber's total material mass.


# %%
def analytical_observations():
    """Generate noise-free readings using the closed-form chain solution."""
    k1, k2 = 0.5, 1.3
    a = np.exp(-k1 * TIMES)
    intermediate = k1 * (a - np.exp(-k2 * TIMES)) / (k2 - k1)
    return 1.0 - a - intermediate, np.ones_like(TIMES)


def concentrations(network, rates, times):
    """Propagate candidate reactions from a one-unit precursor injection."""
    changes = np.asarray(network)
    generator = (changes.T * rates) @ np.maximum(-changes, 0)
    return np.stack([expm(t * generator)[:, 0] for t in times])


# %% [markdown]
# ## Fit each network under each measurement
#
# The objective measures squared concentration residuals relative to a
# common 0.015 µmol/m³ scale. Each rate has bounds of 0–3 s⁻¹. Two
# starts and a common iteration budget give each network the same fitting
# opportunity. This small linear benchmark uses numerical derivatives of
# the objective. The coupled chamber workflow also accepts loss gradients
# from Warp. Compare the largest residual in physical units alongside
# the fitted loss; the weighting scale here describes a hypothetical
# instrument, while the generated observations contain no added noise.


# %%
def fit(network, channel, observed):
    """Fit rates to product concentration or the summed inventory."""

    def evaluate(rates, gradient):
        values = concentrations(network, rates, TIMES)
        predicted = values[:, 2] if channel == "product" else values.sum(axis=1)
        residual = (predicted - observed) / SIGMA
        return float(residual @ residual / 2), None

    size = len(network)
    return fit_experiments(
        [ExperimentLoss(tuple(range(size)), evaluate)],
        [(0.0, 3.0)] * size,
        [np.linspace(0.25, 1.0, size), np.linspace(1.5, 0.3, size)],
        gradient=False,
    )


def simulate():
    """Compare distinguishability with the two synthetic instruments."""
    product, total = analytical_observations()
    search = candidates()
    if search.limit is not None or len(search.networks) != 2:
        raise AssertionError("Expected both feasible networks")
    curve_time = np.linspace(0.0, 8.0, 100)
    report = dict(
        time=TIMES.tolist(),
        curve_time=curve_time.tolist(),
        product=product.tolist(),
        total=total.tolist(),
    )
    for network in search.networks:
        label = "direct" if len(network) == 1 else "chain"
        for channel, observed in (("product", product), ("total", total)):
            fitted = fit(network, channel, observed)
            curve = concentrations(network, fitted.parameters, curve_time)
            signal = curve[:, 2] if channel == "product" else curve.sum(axis=1)
            report[f"{label}_{channel}"] = signal.tolist()
            report[f"{label}_{channel}_loss"] = fitted.loss
            report[f"{label}_{channel}_rates"] = fitted.parameters.tolist()
    if (
        report["direct_product_loss"] <= TIMES.size / 2
        or report["chain_product_loss"] >= 1e-7
    ):
        raise AssertionError(
            "The product trace should resolve the hidden delay"
        )
    np.testing.assert_allclose(
        [report["direct_total_loss"], report["chain_total_loss"]], 0, atol=1e-18
    )
    np.testing.assert_allclose(
        sorted(report["chain_product_rates"]), [0.5, 1.3], rtol=1e-5
    )
    return report


# %% [markdown]
# ## Compare the instruments
#
# The left panel shows the early product lag and the fitted direct
# conversion. The right panel shows the same candidate networks refitted
# using total mass alone. Line styles and an open marker identify the two
# predictions and the analytical measurements, including where they overlap.


# %%
def plot_results(report):
    """Show product discrimination and the ambiguity of total mass."""
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.8), sharey=True, constrained_layout=True
    )
    for ax, channel, title in zip(
        axes,
        ("product", "total"),
        ("Product P", "Total A + I + P"),
        strict=True,
    ):
        for label, name, style in (
            ("direct", "Direct step", "-"),
            ("chain", "Hidden chain", "--"),
        ):
            ax.plot(
                report["curve_time"],
                report[f"{label}_{channel}"],
                style,
                lw=2.4,
                label=name,
            )
        ax.plot(
            report["time"],
            report[channel],
            "o",
            markerfacecolor="none",
            markeredgecolor="black",
            label="Synthetic readings",
            zorder=3,
        )
        ax.set(xlabel="Time (s)", title=title, ylim=(-0.05, 1.13))
    axes[0].set_ylabel("Concentration (µmol/m³)")
    axes[1].legend(loc="center")
    return fig


if __name__ == "__main__":
    plot_results(simulate())
    plt.show()

# %% [markdown]
# Product appearance distinguishes the hidden chain at this sampling cadence:
# the slope of P starts at zero in the chain and is positive in the fitted
# direct reaction. The fitted two-step rates are 0.50 and 1.30 s⁻¹.
# The direct step gives a weighted squared-error loss of 68.5 across nine
# readings. The two-step model reproduces the trace well within the
# stated instrument scale. A conserved total gives identical readings for every
# allowed rate in either network. Even the product curve is unchanged when
# we exchange the two rates within the chain. Measuring precursor decay
# would help assign a rate to the first step. Assigning a chemical identity
# to I requires further evidence about the products or their composition.
