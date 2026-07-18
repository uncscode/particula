# Open Questions

- [x] Which exact two-way masks are part of the first executable support matrix?
  - Resolved 2026-07-16: register all six pairs after their owning tracks ship:
    masks `3`, `5`, `9`, `6`, `10`, and `12`, using canonical bits Brownian `1`,
    charged hard-sphere `2`, SP2016 `4`, and ST1956 `8`. Also retain all four
    singleton rows and register the full four-way mask `15`.
- [x] Are three-way masks deliberately unsupported in E5-F6 or should the
  capability matrix include all non-empty subsets?
  - Resolved 2026-07-16: masks `7`, `11`, `13`, and `14` are deliberately
    unsupported. E5 validates all two-way rows and the full four-way row without
    claiming unrequested three-way coverage.
- [x] Should an unused turbulent dissipation/fluid-density argument be rejected
  for a mask without turbulent shear?
  - Resolved 2026-07-18 by #1357: no. P1's enabled-bit preflight does not inspect
    turbulent arguments for non-turbulent masks; turbulent rows validate them
    before any downstream runtime work.
- [x] Is `sum(component_majorants)` too conservative for the bounded trial cap
  in realistic four-way fixtures?
  - Resolved 2026-07-16: use the sum as the initial proved bound and require
    approved fixtures not to hit the trial cap. If measurement shows cap binding,
    replace it with the exhaustive maximum of the summed total pair rate, not an
    unproved heuristic.
- [x] How should device code surface a material `total_rate > total_majorant`
  violation without host synchronization?
  - Resolved 2026-07-16: a preflight device scan writes an internal per-box
    status buffer that is read before RNG/merge launches. Permit a ratio clamp
    only within `8 * eps * max(abs(rate), abs(majorant), tiny)`; raise on larger
    exceedance. The status buffer is not public API.
