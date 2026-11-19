# %% [markdown]
# # What passes through a 100 nm mobility channel?
#
# In the NaCl chamber experiment of [Cai et al. (2022)](https://doi.org/10.5194/acp-22-11529-2022),
# the collectors were size-selected before they passed through a ⁶³Ni bipolar
# neutralizer. Two different physical diameters can pass the same classifier
# setting if their charges differ. Subsequent ion attachment changes the charge
# while preserving the selected diameter.
#
# ## Estimate the neutralizer outlet fractions
#
# The Wiedensohler approximation gives a separate fit for each signed charge.
# Use metres for diameter and elementary charges for charge. The raw fractions
# below are evaluated independently. The population model in the
# [study record](../../../../contribute/validation/coagulation/nacl-scavenging.md)
# also represents higher states and normalizes before drawing particles.

# %%
import matplotlib.pyplot as plt
import numpy as np
from particula import gas, particles, plotting
from scipy.optimize import brentq

plt.style.use("particula.default")

for diameter_nm, states in [(4, [-1, 0, 1]), (100, [-2, -1, 0, 1, 2])]:
    fractions = particles.get_bipolar_charge_probability(
        diameter_nm * 1e-9, states
    )
    print(f"{diameter_nm} nm")
    for state, fraction in zip(states, fractions, strict=True):
        print(f"  {state:+d} e   {fraction:7.3%}")

# %% [markdown]
# ## Find the true diameters admitted on each charge branch
#
# Electrical mobility has magnitude $|z|e B(d)$, where $B(d)$ is the
# slip-corrected aerodynamic mobility. The elementary charge cancels when
# two mobilities are equal. Solve for the equivalent physical diameter at
# the center of each channel in air at 293.15 K and 101325 Pa.

# %%
viscosity = gas.get_dynamic_viscosity(293.15)
mean_free_path = gas.get_molecule_mean_free_path(
    temperature=293.15, pressure=101325.0, dynamic_viscosity=viscosity
)


def mobility(diameter_nm):
    """Return slip-corrected mechanical mobility in seconds per kilogram."""
    radius = diameter_nm * 0.5e-9
    slip = particles.get_cunningham_slip_correction(mean_free_path / radius)
    return particles.get_aerodynamic_mobility(radius, slip, viscosity)


def equivalent_diameter(nominal, state):
    """Solve for equal electrical mobility on a positive charge branch."""
    return brentq(
        lambda d: state * mobility(d) - mobility(nominal), nominal, 4 * nominal
    )


for target in (4, 100):
    centers = [equivalent_diameter(target, z) for z in (1, 2, 3)]
    print(f"{target} nm channel, +1/+2/+3 centers (nm): {np.round(centers, 2)}")

# %% [markdown]
# ## Compare the steady count retrieval with the measurements
#
# The solid curves below load the archived five-size study results. The
# published-attraction calculation anchors the collision rate at a 100 nm
# collector to the authors' curve and uses the model for its dependence on
# collector size and charge. At 4 nm, larger transmitted collectors can
# account for the residual under the assumed generator spectrum. The same
# assumptions give higher rates than measured at 3.6 and 3.7 nm.

# %%
rates = np.genfromtxt(
    "docs/data/nacl-scavenging/comparison.csv", names=True, delimiter=","
)
capture = np.genfromtxt(
    "docs/data/nacl-scavenging/capture-steady.csv",
    names=True,
    delimiter=",",
    dtype=None,
    encoding="utf-8",
)
full = capture[(capture["case"] == "reference") & (capture["model"] == "full")]
np.testing.assert_array_equal(full["target_nm"], rates["d_nm"])
fig, ax = plotting.subplots()
for field, label, style in [
    ("measured_cm3_s", "Cai measurements", "ko"),
    ("native_steady_cm3_s", "Charged hard spheres", "v-"),
    ("native_plus_vdw_steady_cm3_s", "Earlier additive dispersion", "s-"),
    ("anchored_published_cm3_s", "Published attraction with transfer", "^-"),
]:
    ax.plot(rates["d_nm"], rates[field] * 1e7, style, label=label)
ax.plot(
    full["target_nm"],
    full["apparent_cm3_s"] * 1e7,
    "D--",
    color=plotting.COLORS["berry"],
    label="Dielectric spheres, hard-sphere tail",
)
ax.set(
    xlabel="Nominal tracer diameter (nm)",
    ylabel="Capture coefficient (10⁻⁷ cm³/s)",
)
plotting.legend(ax, frameon=False)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Which particle carries the charge?
#
# Consider one 4 nm tracer and one 100 nm collector at the same temperature,
# pressure and material density. The left panel holds dispersion fixed and
# turns dielectric response on and off. A charge on the small particle can
# polarize the larger partner. Reversing which particle carries the charge
# has a much smaller effect. Compare these pair rates with how many such
# encounters the neutralizer and classifiers actually supply.
#
# The right panel shows the fraction of possible encounters outside the
# kernel's supported domain in the steady chamber and the fraction of the
# true capture rate assigned to those pairs by the hard-sphere assumption.
# Both fractions use the loaded tracer and collector populations.

# %%
pairs = np.genfromtxt(
    "docs/data/nacl-scavenging/capture-pairs.csv", names=True, delimiter=","
)
probe = pairs[pairs["target_nm"] == 4]
fig, (left, right) = plotting.subplots(1, 2)
labels = [
    f"({int(a):+d}, {int(b):+d})"
    for a, b in zip(
        probe["tracer_charge"], probe["collector_charge"], strict=True
    )
]
index = np.arange(len(probe))
for offset, column, label, color in [
    (-0.19, "dispersion_only", "Dispersion + bare charge", "#42838a"),
    (0.19, "full", "With dielectric response", "#955278"),
]:
    left.bar(
        index + offset, probe[column] * 1e7, 0.38, label=label, color=color
    )
left.set(
    xticks=index,
    xticklabels=labels,
    xlabel="(Tracer charge, collector charge) in e",
    ylabel="Pair rate (10⁻⁷ cm³/s)",
    title="4 nm tracer and 100 nm collector",
)
plotting.legend(left)
x = np.arange(len(full))
right.bar(
    x - 0.19,
    100 * full["excluded_pairs"],
    0.38,
    label="Pair fraction",
    color="#809189",
)
right.bar(
    x + 0.19,
    100 * full["tail_bare_cm3_s"] / full["true_capture_cm3_s"],
    0.38,
    label="Share of capture rate",
    color="#c69250",
)
right.set(
    xticks=x,
    xticklabels=[f"{d:g}" for d in full["target_nm"]],
    xlabel="Nominal tracer diameter (nm)",
    ylabel="Outside model domain (%)",
    title="Pairs assigned the hard-sphere rate",
)
plotting.legend(right)
plotting.fit(fig)
plt.show()

# %% [markdown]
# In the recorded 4 nm scan, including polarization changes the apparent rate
# from 1.0651 to 1.0658 × 10⁻⁷ cm³/s. The measured rate was 1.76 × 10⁻⁷.
# The direct force on some charged tracers is substantial, but the mostly
# neutral tracer population gives those pairs little weight. A measurement
# of the collector sizes downstream of the classifier would constrain the
# contribution from the large collectors outside the qualified size range.

# %% [markdown]
# ## How large a rate could settling and shear add?
#
# Compare each increment with the difference between the measurement and the
# full dispersion and polarization model. Use a logarithmic vertical scale
# because the two quantities differ by several orders of magnitude. The
# strongest shear case assumes dissipation of 10 m²/s³ as a sensitivity test.
# The paper reports a low-speed fan but gives no measured dissipation rate.

# %%
other = np.genfromtxt(
    "docs/data/nacl-scavenging/other-kernels-steady.csv",
    names=True,
    delimiter=",",
    dtype=None,
    encoding="utf-8",
)
reference = other[other["case"] == "reference"]
baseline = reference[reference["model"] == "full"]
fig, ax = plotting.subplots(size="wide")
ax.semilogy(
    baseline["target_nm"],
    baseline["measured_cm3_s"] - baseline["apparent_cm3_s"],
    "ko-",
    label="Measured minus modeled",
)
for model, label, style in [
    ("full_settling", "Added by differential settling", "s--"),
    ("full_shear_0.1", "Added by shear at 0.1 m²/s³", "^:"),
    ("full_settling_shear_10", "Added by settling + shear at 10 m²/s³", "D-"),
]:
    selected = reference[reference["model"] == model]
    ax.semilogy(
        selected["target_nm"],
        selected["difference_from_full_cm3_s"],
        style,
        label=label,
    )
ax.set(
    xlabel="Nominal tracer diameter (nm)",
    ylabel="Apparent capture-rate\ndifference (cm³/s)",
)
plotting.legend(ax)
plotting.fit(fig)
plt.show()

# %% [markdown]
# The largest mechanical increment falls several orders of magnitude below
# the residual at every size. The alternative Gatti–Kortshagen thermal fit
# gets closer, but the increase also appears when both collision charges
# are set to zero. The [validation record](../../../../contribute/validation/coagulation/nacl-scavenging.md)
# reports both controls along with the cloud-turbulence extrapolation.

# %% [markdown]
# ## What do individual particles do under the recorded SMPS settings?
#
# Compare the seven measured channel fractions with the simulated signals
# at injection. The second tracer distribution adjusts one size parameter
# to match the mean detected channel in the pooled scans. The right panel
# shows capture events per detector-weighted exposure in 300 s GPU pulses,
# with 95% Monte Carlo intervals across four seeds at a 2.5 s time step.

# %%
pulse = np.genfromtxt(
    "docs/data/nacl-scavenging/pulse-kernels-summary.csv",
    names=True,
    delimiter=",",
    dtype=None,
    encoding="utf-8",
)
spectrum = np.genfromtxt(
    "docs/data/nacl-scavenging/pulse-smps-spectrum.csv",
    names=True,
    delimiter=",",
    dtype=None,
    encoding="utf-8",
)
inlet = spectrum[
    (spectrum["model"] == "full")
    & (spectrum["dt_s"] == 2.5)
    & (spectrum["time_s"] == 0)
    & (spectrum["loading_cm3"] > 0)
]
readout = pulse[(pulse["readout"] == "archived_scan") & (pulse["dt_s"] == 2.5)]
fig, (channels_ax, rates_ax) = plotting.subplots(1, 2, size="wide")
data = inlet[inlet["source"] == "reference"]
channels_ax.plot(
    data["channel_nm"],
    100 * data["observed_pooled_fraction"],
    "ko-",
    label="Measured, 103 scans",
)
models = ["bare", "dispersion_only", "full", "full_settling_shear_10", "gatti"]
for source, label, color, marker, shift in [
    ("reference", "Reference source", plotting.COLORS["teal"], "o", -0.1),
    (
        "smps_conditioned",
        "Fit to mean channel",
        plotting.COLORS["rust"],
        "s",
        0.1,
    ),
]:
    subset = inlet[inlet["source"] == source]
    channels_ax.plot(
        subset["channel_nm"],
        100 * subset["signal_fraction"],
        marker=marker,
        color=color,
        label=label,
    )
    subset = readout[readout["source"] == source]
    rows = [subset[subset["model"] == model][0] for model in models]
    rates_ax.errorbar(
        np.arange(len(rows)) + shift,
        [r["beta_cm3_s"] * 1e7 for r in rows],
        yerr=[r["mc95_halfwidth"] * 1e7 for r in rows],
        fmt=marker,
        color=color,
        capsize=3,
        label=label,
    )
channels_ax.set(
    xlabel="SMPS channel diameter (nm)",
    ylabel="Share of seven-channel signal (%)",
)
plotting.legend(channels_ax)
rates_ax.axhline(
    1.76, color="black", linestyle="--", label="Steady measurement"
)
rates_ax.set(
    ylabel="Pulse capture coefficient\n(10⁻⁷ cm³/s)",
    xticks=np.arange(5),
    xticklabels=[
        "Bare",
        "vdW",
        "vdW +\ndielectric",
        "Plus settling\nand shear",
        "Gatti",
    ],
)
rates_ax.tick_params(axis="x", labelsize=12)
plotting.legend(rates_ax)
plotting.fit(fig)
plt.show()

# %% [markdown]
# Giving the larger tracer channels more weight lowers the pulse capture
# coefficient. Adding settling and shear leaves the sampled capture counts
# unchanged in the matched 4 nm runs. The spectra combine 103 scans spanning
# different chamber conditions. They constrain the shape seen by the instrument;
# the true collector size distribution and the experimental blank-to-loaded
# retrieval require separate information.
