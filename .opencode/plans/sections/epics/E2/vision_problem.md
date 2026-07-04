## Vision and Problem

Epic A establishes the data-model and numerical foundations needed for
reliable multi-box CPU/GPU particle simulations in particula.

### Problems Today

1. **Environment state has no authoritative container.** Temperature,
   pressure, humidity, and saturation state are still passed mostly as
   scalars or legacy `Atmosphere` values, while particle and gas state have
   data-oriented containers.
2. **Gas CPU/GPU schemas have drifted.** `GasData` owns names, molar mass,
   concentration, and partitioning, while `WarpGasData` drops names, converts
   partitioning to `int32`, and adds vapor pressure that is not preserved on
   round trip.
3. **Kernel APIs mix scalar and batched assumptions.** GPU condensation and
   coagulation currently accept scalar temperature and pressure even though
   particles, gas, and volume can be per-box.
4. **Numerical choices need evidence.** Current mass representation and
   explicit condensation stepping are plausible reference choices, but the
   precision and stiffness tradeoffs must be characterized before downstream
   GPU roadmap work depends on them.
5. **Support boundaries are not explicit enough.** Data containers can model
   multi-box state, but several CPU dynamics paths intentionally remain
   single-box or legacy-compatible.

### The Vision

After E2 ships, particula has a documented foundation for data-oriented
simulation state:

- Particle, gas, and environment containers have clear ownership rules and
  shape conventions.
- CPU and Warp containers round-trip through explicit transfer helpers with
  tested semantics.
- Existing scalar GPU kernel calls remain compatible while per-box environment
  state has an incremental migration path.
- Numerical studies provide evidence-backed recommendations for mass
  representation and condensation integration foundations.
- Documentation and examples explain what is supported today, what is limited,
  and how downstream roadmap epics should build on this base.

### Why Now

The data-oriented GPU roadmap already has particle and gas container pieces in
place. Environment state, schema drift, and numerical integration questions are
now the highest-risk blockers for later multi-box GPU features. Resolving them
early prevents downstream API churn and makes later physics work testable.
