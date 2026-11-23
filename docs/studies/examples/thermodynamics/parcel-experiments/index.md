# Parcel replay and exchange

Replay a recorded warm-cloud ascent to measure how sampling affects droplet growth.
Then follow organic vapor and individual settling particles across shared parcel faces.
The lofting study releases the same 1,024 individual particles into three layouts
under a steady wind and updraft. Follow four sizes of salt, organic, and mixed
particles through a vertical stack, around a periodic horizontal row, and across
a grid traveling with the air. The saved histories show how many particles and
how much of each species reach each parcel group or leave the domain.

For the model assumptions, start with [parcel theory](../../../../theory/thermodynamics/parcel-dynamics.md). To build a trace or choose a transport mode, follow [Run a parcel experiment](../../../../guide/processes/parcels.md). The ascent replay and two-cell exchange studies also measure sensitivities on CUDA.

<!-- STUDY_LIST_START -->
- [Can a recorded ascent reproduce droplet growth and close two gas budgets?](parcel_replay.py)
- [Water and organic uptake in a rising parcel](aiomfac_multivapor_parcel.py)
- [What crosses a shared parcel face?](parcel_cluster.py)
- [Where does a particle group travel in wind and updraft?](cluster_lofting.py)
<!-- STUDY_LIST_END -->
