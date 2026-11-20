# CUDA-resident material solvers and respiratory refinement

The comparison used an NVIDIA RTX 5060 at revision `9c548d9d617a93cd23a45773426f751e9e6ed9ec`.

The complete 1,000-particle, twelve-material respiratory example ran to 20 s
on that revision in **286.82 s**. Its separate coarse/fine assessment completed
all **201 observations** in **1,229.39 s**, retaining the original inventory
assertions. The fine run used 100-fold tighter relative and absolute tolerances,
half the maximum step and half the observation interval. Both prospective
0.1% consistency targets passed:

- Maximum per-particle relative radius difference: **3.92121347e-6**.
- Maximum per-species phase-inventory difference: **3.22563458e-8** of that
  species' initial total inventory.
- Coarse: **8,921 accepted**, one rejected step; fine: **41,841 accepted**,
  one rejected step. Neither run had domain rejections.

The complete repository report at
`docs/data/numerics/resident-respiratory-refinement.json` retains every
observation, material identity, controls, source hashes and work counts. These
elapsed times include concurrent GPU work and are not isolated performance
benchmarks. This is same-method numerical consistency evidence;
it does not independently validate the chemistry or correct the example's
previously documented chemical-name matches. The refreshed publication records
CUDA execution, fixed revision and source hashes. The cyclic chamber has
completed its 600 s trajectory and exact checkpoint continuation; the first
publication failed during plot-data serialization, as recorded below.
