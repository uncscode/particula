# Dependency Map

## Inbound

Existing v0.2.13 data containers, process-owned scientific configurations,
supported CPU behavior and regression tests provide the migration baseline.
Existing GPU conversion, resident state and execution contracts are protected
boundaries. Earlier GPU roadmap epics provide context, not E9 parentage or
new measurement prerequisites. No new external dependency is required.

## Strict sequence

`E9-M1 -> E9-M2 -> E9-M3 -> E9-M4 -> E9-M5 -> E9-M6`

Each arrow means **completed and validated**, not merely drafted, started or
merged with tests pending. No parallel track implementation is permitted.

| Handoff | Required artifact |
|---|---|
| M1 → M2 | Approved units/alignment/ownership contract and accessor test evidence |
| M2 → M3 | Data-native construction and replacement contract with identity/rejection evidence |
| M3 → M4 | Native scientific process behavior and explicit adjacent wall-loss results |
| M4 → M5 | Composed CPU workflow and adapter integration evidence |
| M5 → M6 | Supported example/notebook inventory, execution results and strict documentation build |
| M6 → release | Zero remaining supported legacy consumers, compatibility inventory cleared, full validation and release notes |

Upstream contract changes reopen affected gates. Child planners must encode
this same dependency chain in child metadata; E9's drafting scope does not
edit those records. The epic milestones summarize these gates rather than
authorizing parallel work.

## Outbound

v0.3.0 release readiness and subsequent Epic I work depend on this migration.
This plan neither implements Epic I nor claims release approval.
