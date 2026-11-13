# Dependencies

## Upstream

- **E6-F5 is deferred and mandatory for composition.** It owns authoritative
  slot discovery, free-index classification, and activation, including the
  active/free truth table, ascending free indices, exact int32 counts,
  fixed-shape ownership, and failure-before-mutation contract.
- **E6-F6-P5 (#1426) remains deferred pending E6-F5.** No P5 implementation
  is present; resume only after E6-F5 supplies the authoritative
  discovery-to-activation boundary. E6-F6's shipped primitives must not
  duplicate that boundary or imply an integrated runtime policy loop.
- Shipped E5 coagulation supplies fixed-slot merge/deactivation, charge
  conservation, persistent-state, and Warp validation conventions.
- `ParticleData`/`WarpParticleData`, NumPy, and Warp are runtime foundations.
  Warp CPU is required validation evidence; CUDA remains optional.

## Downstream

- **E6-F7** owns downstream CPU source construction and gas/particle-inventory
  mutation after E6-F5 and E6-F6 are available.
- **E6-F8** owns downstream direct Warp nucleation after E6-F5, E6-F6, and
  E6-F7.
- **E6-F9** owns downstream integrated direct-process sequencing and complete
  exhaustion validation; E6-F6 does not provide that orchestration.
- Epic G may schedule these primitives later but may not change E6-F6 policy
  defaults or ownership implicitly.

## Sibling Boundaries

- E6-F5 classifies/activates slots but never chooses an exhaustion policy.
- E6-F7/E6-F8 own nucleation physics, source construction, and gas-inventory
  limitation. E6-F6 supplies only planning and direct fixed-capacity primitives;
  it does not discover/activate slots or finalize source records.
- E6-F3/E6-F4 can create free slots through wall loss but do not invoke E6-F6.

## Phase Ordering

P1 freezes the contract before P2 CPU resampling; P3 ports it to Warp; P4 adds
the optional scaling path; P5 remains deferred pending E6-F5 composition; P6
supplies a no-public-contract cross-backend conservation matrix; P7 is the
documentation-only phase and adds no pytest module.
