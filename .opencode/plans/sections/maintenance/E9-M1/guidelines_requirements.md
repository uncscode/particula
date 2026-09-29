# Guidelines and Requirements

## Functional requirements

1. Preserve existing container names and array layouts; add distribution_type
   metadata per E9 appendix D1–D2. The minimum contract table
   must record these authorities; B, N, Sp and Sg denote boxes, particles,
   particle species and gas species. Their relationship is not inferred from
   equal widths.

   | Owner / field | Shape | Meaning |
   |---|---|---|
   | ParticleData.masses | (B, N, Sp) | kg per represented particle, per species |
    | ParticleData.concentration | (B, N) | Discrete/PMF bin number per m³; radius PDF dn/dr per m⁴; resolved represented counts in physical V |
   | ParticleData.distribution_type | Scalar metadata | Shared discrete / continuous_pdf / particle_resolved vocabulary |
   | ParticleData.charge | (B, N) | Elementary-charge counts per particle |
   | ParticleData.density | (Sp,) | Material density, kg/m^3 |
    | ParticleData.volume | (B,) | Fixed 1 m³ for PMF/PDF; positive finite physical V for resolved counts |
   | GasData.name / molar_mass / partitioning | Sg / (Sg,) / (Sg,) | Ordered names, kg/mol and boolean participation mask |
   | GasData.concentration | (B, Sg) | Gas mass concentration, kg/m^3, all gas categories |
   | EnvironmentData.temperature / pressure | (B,) | K / Pa |
   | EnvironmentData.saturation_ratio | (B, Sg) | Dimensionless lanes in full gas order |

2. Use the approved explicit process-owned index map (E9 appendix D3). Record
   ordered names and how unnamed particle/environment lanes and every
   species-indexed process parameter are associated. No silent sorting,
   truncation, name-based guessing, or partitioning-mask compaction. Distinguish
   nonpartitioning from absent gas; both gas categories remain in GasData.
   Specify duplicate/missing-name and reordered-configuration handling. If a
   permutation cannot be detected from unnamed arrays, require the caller's
   declared mapping rather than claiming shape validation detects it.
 3. Implement the clarified P1 normalization ledger (which supersedes the
    original D1 proposals). Resolved counts divide by V exactly once; PMF bin
    values and radius PDF are already per-volume. PDF totals use trapezoidal
    integration on an explicit radius grid; never sum slots or divide PDF/PMF
    density again. Only resolved density inputs/increments multiply by V on
    storage. Gas remains kg/m³. Builders require explicit kind and validate
    the representation-specific volume on creation. Direct construction and
    later raw-array mutation use an explicit read-only representation validator;
    new bulk helpers call it before interpretation. Do not add blanket
    constructor coercion or automatic mutation interception.
   Audit every CPU/GPU consumer; old unit-volume agreement is not sufficient.
   Per-particle total_mass is sum_s(m_i,s), not population mass. Distribution
   metadata describes storage; processes own compatible algorithm selection.
4. Reuse `radii`, `total_mass`, `effective_density`, `mass_fractions` and
   `copy()` when sufficient. Add a helper only with a named downstream use.
   Define units, dimensions, physical domain, copy/view identity, allowed
   mutation, empty/zero behavior and rejection semantics for each addition.
   Derived values are computed from current arrays, never stored as competing
   authoritative caches. No physics strategies belong on a container.
5. Specify appendix D4's property-based aggregate access for M2: properties
   return held objects; validated setters retain the supplied object and check
   it against the other two held containers. Coordinated replacement validates
   the complete proposed triple before publishing any reference. Rejection
   preserves old references, their arrays and candidate inputs. Validation must
   not invoke coercing constructors on already-held objects or mutate names.
6. Define structural compatibility versus process physical validation. Direct
   field mutation is allowed and remains subject to process revalidation.
   Single-box process admission remains unchanged; batched container support
   does not authorize multi-box CPU execution. CPU replacement never rebinds
   existing prepared GPU/resident containers or arrays.

## Quality bars

- Independent NumPy scientific oracles, per-species assertions and explicit
  tolerances; non-unit volumes must expose missing/double normalization.
- Each function change ships adjacent `*_test.py` tests in the same PR.
- Public helpers have typed inputs/outputs and Google-style docstrings;
  follow 80-column Ruff formatting and required mypy checks.
- Preserve existing behavioral coverage, protected transfers and exports.
  Inventory temporary compatibility if unavoidable; M6 removes it.

## Constraints

P1 → P2 → P3 → P4 → P5 is serial. Each phase requires its predecessor's
evidence. No sibling implementation starts before the completed-and-validated
M1 gate. No new dependencies or arbitrary coverage thresholds; focused checks
are assertions only and comprehensive evidence uses the untargeted runner.
