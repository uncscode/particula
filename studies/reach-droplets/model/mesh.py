"""Volume-conservative cylindrical cells, physical air fluxes, and finite-beam integrals."""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.special import roots_legendre

R = 0.254
H = 0.495
WIDTH = 0.0054
HEIGHT = 0.0031
DEPTH = 0.00331  # m per micrometre of wet diameter


@dataclass
class Grid:
    nr: int
    nz: int
    na: int

    def __post_init__(self):
        self.dr = R / self.nr
        self.dz = H / self.nz
        self.da = 2 * np.pi / self.na
        self.re = np.linspace(0, R, self.nr + 1)
        self.ze = np.linspace(0, H, self.nz + 1)
        self.rc = (self.re[:-1] + self.re[1:]) / 2
        self.zc = (self.ze[:-1] + self.ze[1:]) / 2
        self.ac = (np.arange(self.na) + 0.5) * self.da
        self.shape = (self.nz, self.na, self.nr)
        one = np.diff(self.re**2) / 2 * self.da * self.dz
        self.vol = np.broadcast_to(one, self.shape).ravel().copy()
        self.v = self.vol / self.vol.sum()
        self.n = len(self.v)

    def index(self, z, a, r):
        return (z * self.na + a) * self.nr + r


def along(grid, origin, direction, low, high):
    """Exact lengths of straight line intersections with cylindrical sector walls."""
    x, y = origin
    dx, dy = direction
    edges = [low, high]
    proj = x * dx + y * dy
    normal = x * dy - y * dx
    for r in grid.re:
        delta = r * r - normal * normal
        if delta > 0:
            root = np.sqrt(delta)
            edges.extend([-proj - root, -proj + root])
    for angle in np.arange(grid.na) * grid.da:
        c, s = np.cos(angle), np.sin(angle)
        den = dx * s - dy * c
        if abs(den) > 1e-13:
            edges.append((y * c - x * s) / den)
    pts = np.unique(np.clip(edges, low, high))
    mid = (pts[1:] + pts[:-1]) / 2
    xx = x + mid * dx
    yy = y + mid * dy
    rr = np.hypot(xx, yy)
    a = (np.arctan2(yy, xx) % (2 * np.pi) / grid.da).astype(int) % grid.na
    ir = np.minimum((rr / grid.dr).astype(int), grid.nr - 1)
    inside = rr < R
    return np.bincount(
        (a * grid.nr + ir)[inside], weights=np.diff(pts)[inside], minlength=grid.nr * grid.na
    )


def z_weights(grid, z, thickness=0.0, linear=True):
    """Integrate a piecewise linear vertical concentration profile across the receiver."""
    if thickness and not linear:
        return (
            np.maximum(
                0,
                np.minimum(grid.ze[1:], z + thickness / 2)
                - np.maximum(grid.ze[:-1], z - thickness / 2),
            )
            / thickness
        )
    if not linear:
        w = np.zeros(grid.nz)
        w[min(int(z / grid.dz), grid.nz - 1)] = 1
        return w

    def point(height):
        w = np.zeros(grid.nz)
        pos = np.clip(height / grid.dz - 0.5, 0, grid.nz - 1)
        i = int(pos)
        w[i] += 1 - (pos - i)
        w[min(i + 1, grid.nz - 1)] += pos - i
        return w

    if thickness == 0:
        return point(z)
    ends = np.unique(
        np.clip(
            np.r_[grid.zc, z - thickness / 2, z + thickness / 2],
            z - thickness / 2,
            z + thickness / 2,
        )
    )
    return sum(
        (point(a) + point(b)) * (b - a) / (2 * thickness) for a, b in zip(ends[:-1], ends[1:])
    )


def hologram(grid, diam, z=H / 2, axial=0, lateral=0, skew=0, order=10, linear=True):
    # y is in the optical depth direction; x spans the measured horizontal width.
    x, weights = roots_legendre(order)
    zlow = z - HEIGHT / 2
    zhigh = z + HEIGHT / 2
    share = z_weights(grid, z, HEIGHT, linear)
    out = np.zeros((grid.n, len(diam)))
    coverage = []
    for j, d in enumerate(diam):
        length = DEPTH * d
        center = axial + skew * length
        cross = np.zeros(grid.nr * grid.na)
        for q, w in zip(x, weights):
            cross += (
                0.5
                * w
                * along(
                    grid,
                    (lateral + q * WIDTH / 2, 0.0),
                    (0.0, 1.0),
                    center - length / 2,
                    center + length / 2,
                )
            )
        mass = (share[:, None] * cross).ravel() / length
        coverage.append(mass.sum())
        if mass.sum() <= 0:
            raise ValueError("Empty optical footprint")
        out[:, j] = mass / mass.sum()
    return out, np.array(coverage)


def gas_path(grid, z, legacy=False):
    # Figure 1a top view: two shallowly inclined legs from left viewport to a wall mirror.
    # Positions in cylinder radii, abstracted from the schematic; absolute gas path provided by paper.
    if legacy:
        legs = [((-1, 0), (1, 0))]
    else:
        legs = [((-0.99, 0.13), (0.90, -0.28)), ((-0.99, -0.02), (0.90, -0.28))]
    weight = np.zeros(grid.nr * grid.na)
    for a, b in legs:
        a = np.array(a) * R
        b = np.array(b) * R
        diff = b - a
        length = np.linalg.norm(diff)
        weight += along(grid, a, diff / length, 0, length)
    result = z_weights(grid, z)[:, None] * weight[None, :]
    return result.ravel() / result.sum()


def air(grid, run, flow, scales=None):
    """Finite-volume face diffusion plus upwind flux of a closed prescribed circulation."""
    # Characteristic droplet speeds in the comparable apparatus. Mean directional flow fraction is a scenario.
    scale = 0.8 if run == "forced" else 0.1
    if scales is not None:
        u, k = scales
        if k < 0:
            raise ValueError("Diffusivity must be nonnegative")
        flow = "reverse" if u < 0 else "jets"
    velocity = 0.35 * scale
    diffusivity = scale * 0.05 / 3
    if scales is not None:
        velocity *= abs(u)
        diffusivity *= k
    if flow in ("slow", "slow-advection"):
        velocity *= 0.25
    if flow in ("slow", "slow-diffusion"):
        diffusivity *= 0.25
    if flow == "fast":
        velocity *= 2
        diffusivity *= 2
    sign = 1 if flow == "reverse" else -1
    if flow == "exchange":
        velocity = 0.0
    waves = 3 if run == "forced" else 1
    phase = np.pi / 6 if flow == "rotated" else 0.0
    modulation = 0.0 if flow == "roll" or grid.na == 1 else 0.8
    re = grid.re / R
    ze = grid.ze / H
    # Evaluate angular average over each face, giving a resolution-independent streamfunction.
    angular = 1 + modulation * np.cos(waves * (grid.ac - phase)) * np.sinc(
        waves * grid.da / (2 * np.pi)
    )
    psi = (
        sign
        * 0.5
        * velocity
        * R**2
        * grid.da
        * np.sin(np.pi * ze)[:, None, None]
        * (0.6 + ze[:, None, None])
        * angular[None, :, None]
        * (re**2 * (1 - re**2))[None, None, :]
    )
    rows = []
    cols = []
    data = []

    def edge(a, b, adv, conduct):
        # Inputs are volumetric flux and exchange conductance, m3/s. Columns act on cell inventories.
        ab = (conduct + max(adv, 0)) / grid.vol[a]
        ba = (conduct + max(-adv, 0)) / grid.vol[b]
        rows.extend([b, a, a, b])
        cols.extend([a, a, b, b])
        data.extend([ab, -ab, ba, -ba])

    for iz in range(grid.nz):
        for ia in range(grid.na):
            for ir in range(grid.nr):
                a = grid.index(iz, ia, ir)
                if ir + 1 < grid.nr:
                    adv = psi[iz, ia, ir + 1] - psi[iz + 1, ia, ir + 1]
                    area = grid.re[ir + 1] * grid.da * grid.dz
                    edge(a, a + 1, adv, diffusivity * area / grid.dr)
                if iz + 1 < grid.nz:
                    adv = psi[iz + 1, ia, ir + 1] - psi[iz + 1, ia, ir]
                    area = grid.vol[a] / grid.dz
                    edge(a, a + grid.nr * grid.na, adv, diffusivity * area / grid.dz)
                if grid.na > 1:
                    b = grid.index(iz, (ia + 1) % grid.na, ir)
                    area = grid.dr * grid.dz
                    azimuth = 0.35 * velocity * (grid.rc[ir] / R) if flow == "swirl" else 0.0
                    edge(a, b, azimuth * area, diffusivity * area / (grid.rc[ir] * grid.da))
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(grid.n, grid.n)).tocsc()
    assert np.max(abs(matrix.sum(0))) < 1e-8 and np.max(abs(matrix @ grid.v)) < 1e-12
    return matrix
