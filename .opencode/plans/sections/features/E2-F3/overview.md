# Overview

Feature `E2-F3` implements the Warp-side representation of the environment
container introduced by `E2-F2` and adds explicit CPU/GPU conversion helpers.
The track is tied to issue `#1172` / track `T3` in parent epic `E2`.

## Goals

- Add `WarpEnvironmentData` as the GPU mirror of the CPU `EnvironmentData`
  schema, preserving per-box array semantics for temperature, pressure, and
  derived humidity or saturation fields.
- Add explicit `to_warp_environment_data` and `from_warp_environment_data`
  helpers that follow existing `ParticleData` and `GasData` transfer patterns.
- Prove lossless environment round trips on the Warp CPU backend.
- Add CUDA-parametrized coverage that activates only when Warp reports CUDA is
  available.
- Document transfer semantics so simulation code does not acquire hidden
  CPU/GPU synchronization or transfer behavior.

## Motivation

The data-oriented GPU roadmap requires per-box environmental state to travel
alongside gas and particle data before kernels can migrate from scalar
temperature and pressure inputs to batched environment arrays. This feature is
the bridge between the CPU environment schema from `E2-F2` and later kernel
migration tracks (`E2-F5+`).

## Parent and sibling context

- Parent epic: `E2` (Data-model and numerical foundations v2).
- Upstream dependency: `E2-F2` / `T2`, which defines the CPU
  `EnvironmentData` schema and validation boundary.
- Related completed tracks: `E2-F1` established CPU data-container patterns;
  `E2-F2` provides the environment container this feature mirrors.
- Downstream tracks consume the conversion helpers rather than performing
  ad-hoc Warp transfers inside kernels.
