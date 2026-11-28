# Particula native atmospheric reactions

The [snapshot manifest](snapshots.json) identifies the source configuration,
input digests and citations for each conversion. Each JSON line contains one
Particula gas mechanism with its own rate units, species compositions and
source reaction ID. The release ZIP packages these files from the repository.

| Native records | Count | Upstream reference |
| --- | ---: | --- |
| [WACCM TSMLT1](waccm.jsonl) | 39 | [NCAR Whole Atmosphere Working Group](https://www.cesm.ucar.edu/working-groups/whole-atmosphere/code) |
| [MCM v3.3.1 α-pinene](mcm.jsonl) | 95 | [University of York mechanism exporter](https://www.mcm.york.ac.uk/MCM/export) |
| [MCM v3.3.1 full export](mcm-full.jsonl) | 2,153 | [MCM archive](https://www.mcm.york.ac.uk/MCM/about/archive) · [How to cite the mechanism and its construction protocols](https://www.mcm.york.ac.uk/MCM/about/cite) |
| [E3SM](e3sm.jsonl) | 4 | [E3SM v3.0.0](https://github.com/E3SM-Project/E3SM/releases/tag/v3.0.0) |
| [WRF-Chem](wrf-chem.jsonl) | 4 | [WRF v4.6.1](https://github.com/wrf-model/WRF/releases/tag/v4.6.1) |
| [CMAQ CRACMM2](cmaq.jsonl) | 4 | [EPA CMAQ mechanism](https://github.com/USEPA/CMAQ/tree/5.5/CCTM/src/MECHS/cracmm2) |
| [CAMx](camx.jsonl) | 6 | [CAMx source and documentation](https://www.camx.com/download/source/) |
| [PMCAMx gas chemistry](pmcamx.jsonl) | 6 | [Archived PMCAMx-SR sources](https://doi.org/10.5281/zenodo.4071362) |

The ten source configurations contain 19,655 entries. The table counts 2,311
individually converted channels, with overlap across configurations. The two
PMCAMx SOAP configurations have zero native gas channels. Their yields require
host oxidants and partitioning. Seven NO + O₃ records set
`requires_closed_oxygen` and require the study to include an evolving O₂
product. Catalog export enforces this choice through `--close-oxygen`.

For offline search, selection locks and export, see the
[gas chemistry guide](../docs/guide/processes/gas-chemistry.md#export-a-selected-gas-reaction).
