# Guidelines and Requirements

## Functional requirements

1. Preserve existing container names and schemas. The minimum contract table
   must record these authorities; B, N, Sp and Sg denote boxes, particles,
   particle species and gas species. Their relationship is not inferred from
   equal widths.

   | Owner / field | Shape | Meaning |
   |---|---|---|
   | ParticleData.masses | (B, N, Sp) | kg per represented particle, per species |
   | ParticleData.concentration | (B, N) | Raw weights/counts or number density according to the audited convention |
   | ParticleData.charge | (B, N) | Elementary-charge counts per particle |
   | ParticleData.density | (Sp,) | Material density, kg/m^3 |
   | ParticleData.volume | (B,) | Represented box volume, m^3 |
   | GasData.name / molar_mass / partitioning | Sg / (Sg,) / (Sg,) | Ordered names, kg/mol and boolean participation mask |
   | GasData.concentration | (B, Sg) | Gas mass concentration, kg/m^3, all gas categories |
   | EnvironmentData.temperature / pressure | (B,) | K / Pa |
   | EnvironmentData.saturation_ratio | (B, Se) | Dimensionless species lanes; Se mapping must be frozen in P1 |

2. P1 must approve one explicit mapping: a shared full-width order, or an
   explicit process-owned index map that preserves current layouts. Record
   ordered names and how unnamed particle/environment lanes and every
   species-indexed process parameter are associated. No silent sorting,
   truncation, name-based guessing, or partitioning-mask compaction. Distinguish
   nonpartitioning from absent gas; both gas categories remain in GasData.
   Specify duplicate/missing-name and reordered-configuration handling. If a
   permutation cannot be detected from unnamed arrays, require the caller's
   declared mapping rather than claiming shape validation detects it.
3. Freeze a distribution-specific normalization ledger before helper design.
   For count/weight storage w, physical number density is w/V; for density
   storage c, it is c. Never divide twice. Per-species population mass density
   is sum_i(m_i,s * c_i); extensive mass is that density times V. Gas extensive
   mass is gas.concentration_s * V. Per-particle total_mass is sum_s(m_i,s),
   not any of these population quantities. Resolve the current documentation
   versus legacy-getter discrepancy with consumer evidence, not a storage
   reinterpretation. Keep any required interpretation explicit and process-owned.
4. Reuse `radii`, `total_mass`, `effective_density`, `mass_fractions` and
   `copy()` when sufficient. Add a helper only with a named downstream use.
   Define units, dimensions, physical domain, copy/view identity, allowed
   mutation, empty/zero behavior and rejection semantics for each addition.
   Derived values are computed from current arrays, never stored as competing
   authoritative caches. No physics strategies belong on a container.
5. Specify aggregate access for M2: whole-container getters return the held
   objects; validated individual setters retain the supplied object and check
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
