"""Read a native numeric netCDF4/HDF5 field using the parent reader."""

import importlib.util
from pathlib import Path

source = Path(__file__).resolve().parents[2] / "extract.py"
spec = importlib.util.spec_from_file_location("parent_netcdf", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
field = module.read_field
