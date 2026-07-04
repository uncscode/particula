# Risk Register

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| Ownership decisions remain ambiguous for `vapor_pressure` or volume. | Downstream tracks may implement incompatible schemas. | Require explicit owner/round-trip entries in the decision table before E2-F2/E2-F4 implementation. | E2-F1 implementer |
| Shape docs imply multi-box process support where CPU strategies still require one box. | Users and implementers may expect unsupported behavior. | Add a support-boundary note distinguishing container capacity from process API support. | E2-F1 implementer |
| GPU metadata loss for names or vapor pressure is under-documented. | Silent loss during CPU/GPU conversion can cause scientific or debugging errors. | Document lossy round-trip behavior and required metadata sidecars or return arguments. | E2-F1 implementer |
| Decision record becomes stale as downstream tracks implement containers. | Later work diverges from the foundation. | Require sibling tracks to update the decision record or link final implementation notes. | E2 feature owners |
| Documentation-only work lacks verification. | Reviewers may miss mismatches with code. | Cross-check against existing tests and run docs/link validation where available. | Phase implementer |

## Highest-Priority Mitigations

1. Publish the ownership table before environment implementation begins.
2. Treat lossy CPU/GPU conversion semantics as first-class documented behavior.
3. Keep open questions narrow and assign each to a downstream track or owner.
