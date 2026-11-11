# Scope

M2 implements native Aerosol construction/access/replacement, native
AerosolBuilder, and useful native mass/radius/bin/resolved initialization.
Production changes are bounded to these construction surfaces, minimal exports,
and explicit legacy import routing. Tests ship with their owning phases.

`particula._legacy_aerosol` retains the old aggregate and builder, with their
historical aliasing and mutation behavior. The approved temporary seam is
inventoried in `legacy_consumers.md`; M6 removes it after named consumers
migrate. There is no native atmosphere emulation or duplicate authoritative
state. Example changes route imports and synchronize paired sources, with
full tutorial migration/execution remaining M5-owned.

The new aggregate runs the existing structural validator followed by particle
representation validation. Unequal particle/gas species counts and all-false
gas masks are valid. Mapping and physical process admission remain downstream.
The delivered M1 fixed-volume PMF/PDF and variable-volume resolved-count rules
supersede original D1 density-times-volume proposals.

Out of scope: scientific process migration (M3), runnable/adapters and resident
composition (M4), broad tutorials (M5), legacy removal/release (M6), new GPU
schema/interpretation, tagged Warp upload, checkpoint metadata, automatic GPU
rebind, hidden transfer/fallback, general multi-box CPU execution and new
sampling physics. Existing untagged transfer and device ownership are preserved.
