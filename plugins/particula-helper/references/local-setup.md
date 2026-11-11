# Local execution
The plugin contains skills and supporting Python scripts, not an MCP server. A compatible host must expose local Python execution and file tools. Uploading a ZIP does not connect web/mobile chat to your computer. No API key or hosted server is needed for these scripts.

Use an existing suitable environment where possible. These native starters require a Particula checkout containing the E9 native
data APIs. They do not run against the older 0.2.x facade-only API. For a new
project, use Python 3.12 or newer and install the selected native checkout:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r /absolute/path/to/particula-helper/requirements.txt
.venv/bin/python -m pip install -e /absolute/path/to/native-particula-checkout
.venv/bin/python /absolute/path/to/particula-helper/scripts/doctor.py
.venv/bin/python /absolute/path/to/particula-helper/scripts/run_example.py --process condensation --output ./runs/condensation-001
```
On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`. Substitute the actual extracted package path. An existing output directory is intentionally rejected; choose a new run name.

The pinned starter depends on Python/platform wheel availability. For another Python/platform use a separately resolved environment and revalidate it; never change the user's research environment silently.

Two starter processes are provided: condensation and Brownian coagulation. Other Particula processes are handled through the design/run skills using verified version-specific code. GPU execution is not tested or enabled by the starter.
