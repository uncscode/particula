# %% [markdown]
# # Figure styles and paper layouts
#
# How can we compare size distributions and show the sign of a model error at
# paper size? These illustrations use analytic curves and fields. They show
# visual encodings and physical dimensions rather than chamber measurements.
#
# ## Set up the figures
#
# Activate the package style for this script. The panel helpers return ordinary
# Matplotlib figures and axes. Change individual artist properties as usual.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm, to_rgb
from particula import plotting

plt.style.use("particula.default")
diameter = np.geomspace(10, 500, 250)


def profile(center):
    """Return a peak-normalized illustrative size distribution."""
    return np.exp(-0.5 * (np.log(diameter / center) / np.log(1.5)) ** 2)


# %% [markdown]
# ## Square panel and inside legend
#
# Plot two median diameters against a log-scaled diameter axis. Their short
# labels fit inside the panel at 12 pt. The dashed and solid strokes also
# distinguish the distributions in monochrome.

# %%
fig, ax = plotting.subplots()
for center in (50, 120):
    ax.semilogx(diameter, profile(center), label=f"{center} nm")
ax.set(xlabel="Particle diameter (nm)", ylabel="Normalized count")
plotting.legend(ax)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Wide panel and outside legend
#
# Use a wider data panel for a four-way comparison. The long labels move the
# legend into a measured right gutter. Each curve still spans a 5 × 3.5 inch
# axes. Use the reported canvas width when choosing a paper column.

# %%
fig, ax = plotting.subplots(size="wide")
for center in (30, 60, 100, 180):
    ax.semilogx(diameter, profile(center), label=f"Median diameter {center} nm")
ax.set(xlabel="Particle diameter (nm)", ylabel="Normalized count")
plotting.legend(ax)
plotting.fit(fig)
print("Finished wide figure (inches):", np.round(fig.get_size_inches(), 2))
plt.show()

# %% [markdown]
# ## Magnitude image
#
# Show a synthetic concentration field evolving over time. The labeled colorbar
# uses a logarithmic scale. Deep teal marks larger positive concentrations.
# Keep the grid off for raster fields and specify the image origin explicitly.

# %%
time = np.linspace(0, 6, 180)
height = np.linspace(0, 1, 160)
peak = 0.25 + 0.48 * time / time.max()
field = 10 + 990 * np.exp(-(((height[:, None] - peak) / 0.14) ** 2))
fig, ax = plotting.subplots()
ax.grid(False)
image = ax.imshow(
    field,
    origin="lower",
    aspect="auto",
    extent=(0, 6, 0, 1),
    cmap=plotting.colormap("magnitude"),
    norm=LogNorm(10, 1000),
)
ax.set(xlabel="Elapsed time (h)", ylabel="Height (km)")
fig.colorbar(image, ax=ax, label="Number concentration (cm⁻³)")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Signed filled contours
#
# Compare the concentration field with the same idealized layer shifted upward.
# Use zero as the reference for the signed difference. Negative contours are
# rust and positive contours are teal. A ratio would instead use one as its
# reference. The caller chooses the levels, normalization, and colorbar units.

# %%
shifted = 10 + 990 * np.exp(-(((height[:, None] - peak - 0.06) / 0.14) ** 2))
difference = shifted - field
limit = 400
fig, ax = plotting.subplots()
ax.grid(False)
contours = ax.contourf(
    time,
    height,
    difference,
    levels=np.linspace(-limit, limit, 17),
    norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
    cmap=plotting.colormap("difference"),
)
ax.set(xlabel="Elapsed time (h)", ylabel="Height (km)")
fig.colorbar(contours, ax=ax, label="Number difference (cm⁻³)")
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Read the curves in grayscale
#
# Render the same four strokes at their grayscale luminance. Color conveys the
# series in the website version, and the dash patterns carry it into print.

# %%
fig, ax = plotting.subplots(size="wide")

for center, color in zip(
    (30, 60, 100, 180), plotting.COLORS.values(), strict=True
):
    # Relative luminance in linear sRGB, converted back to display gray.
    rgb = np.array(to_rgb(color))
    linear = np.where(
        rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
    )
    luminance = linear @ [0.2126, 0.7152, 0.0722]
    gray = 1.055 * luminance ** (1 / 2.4) - 0.055
    ax.semilogx(
        diameter, profile(center), color=str(gray), label=f"{center} nm"
    )
ax.set(xlabel="Particle diameter (nm)", ylabel="Normalized count")
plotting.legend(ax, placement="right")
plotting.fit(fig)
plt.show()

# %% [markdown]
# The square and wide examples preserve the space available for data while
# accommodating labels outside it. Choose the sequential scale for amount and
# the diverging scale when a physical reference divides the quantity into two
# interpretations. To adapt the comparison, replace one analytic curve with
# measurements and add error bars in the same series color.
