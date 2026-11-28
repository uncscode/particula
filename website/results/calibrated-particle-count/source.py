# %% [markdown]
# # When 100 counts per second means 10 or 5 particles per cubic centimetre
#
# A particle counter measures pulses per second. To infer number concentration
# at the inlet, we also need the volume of air sampled each second. At 100
# counts/s, a flow of 0.60 L/min samples 10 cm³/s and gives 10 cm⁻³. Doubling
# that flow gives 5 cm⁻³ at the same detector rate. Which value belongs to a
# reading depends on the calibration in effect when it was taken.
#
# In this five-second synthetic bench run, the flow meter reads 2 V over its
# first three seconds. A calibration takes effect at second 2 and doubles the
# slope relating voltage to flow. The counter and flow meter share an inlet,
# an ambient-volume basis and a UTC clock. We assume one count per detected
# particle and unit detection efficiency. The calibration change is prescribed
# to isolate its effect on the inferred concentration.
#
# Install the checkout with `pip install ".[data]"` and extract the complete
# example ZIP. From the extracted directory, run `python example.py`. The
# download includes both synthetic CSV files, the flow meter's import profile,
# a two-step recipe and the lab's Python module. The lab supplies
# `linear_flow(signal, slope, intercept)`, which computes
# `slope * signal + intercept`. Its `flow_process()` declares voltage in V as
# input and flow in L/min as output. The registry resolves that declaration
# when the recipe asks for `lab.flow.linear`. See the
# [measurement processing guide](../../guide/data-model/measurement-processing.md)
# for the public contracts used by a lab extension.

# %%
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from particula.data import (
    Calibration,
    Deployment,
    OperationRegistry,
    open_measurement,
    run_recipe,
)
from particula.instruments import Instrument, ParticleCounter
from scripts.measurement_flow_lab import flow_process


def data_directory():
    """Find the synthetic records in a checkout or extracted ZIP."""
    relative = Path("docs/data/measurement-processing")
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.append(Path(__file__).resolve().parent)
    for start in starts:
        for parent in (start, *start.parents):
            if (parent / relative / "flow.csv").is_file():
                return parent / relative
    raise FileNotFoundError(f"Extract the complete ZIP containing {relative}")


# %% [markdown]
# ## Assign a calibration to each reading
#
# The first certificate applies from second 0 up to second 2. The second
# starts at second 2. Both name the calibration group `flow-scale`, so the
# same recipe can select the applicable slope by timestamp. The references
# identify the individual certificates when the processed data are reopened.
# Reading the inputs through their instrument objects associates them with
# these deployments before any derived field is calculated.


# %%
def setup_inputs(source):
    """Bind the counter and custom flow meter to the same inlet."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    calibrations = tuple(
        Calibration(
            reference=reference,
            group="flow-scale",
            operation="lab.flow.linear",
            start=start + timedelta(seconds=first),
            stop=start + timedelta(seconds=stop),
            coefficients={"slope": slope, "intercept": 0.0},
            units={"slope": "L/min/V", "intercept": "L/min"},
            source=f"Synthetic bench certificate {reference}",
        )
        for reference, first, stop, slope in (
            ("bench-A", 0, 2, 0.30),
            ("bench-B", 2, 5, 0.60),
        )
    )
    registry = OperationRegistry()
    registry.register(flow_process())
    meter = Instrument(
        source / "flow-profile.yaml",
        deployment=Deployment(
            name="flow",
            inlet="inlet",
            volume_basis="ambient",
            calibrations=calibrations,
        ),
        registry=registry,
    )
    counter = ParticleCounter(
        deployment=Deployment(
            name="counter", inlet="inlet", volume_basis="ambient"
        ),
        registry=registry,
    )
    voltage = meter.read(source / "flow.csv", as_product="voltage")
    rate = counter.read(source / "counter.csv", as_product="rate")
    return registry, meter, counter, voltage, rate


# %% [markdown]
# ## Apply the correction and replay it from a recipe
#
# The Python calls first convert voltage to flow, then divide count rate by
# sample volume per second at matching timestamps. The distributed YAML recipe
# names these same operations, their versions and their input roles. Run it
# with the same raw measurements, then serialize that YAML as JSON and replay
# it to a new directory. Reopen the materialized result and compare every
# concentration, warning and calibration reference with the direct calculation.


# %%
def run_example():
    """Run the direct and prescribed calculations on the bench observations."""
    source = data_directory()
    registry, meter, counter, voltage, rate = setup_inputs(source)
    calibrate = meter.process("lab.flow.linear", version="1").configure(
        calibration="flow-scale"
    )
    flow = calibrate.execute(signal=voltage)
    derive = counter.process(
        "particula.particle_counter.number_concentration", version="1"
    )
    direct = derive.execute(rate=rate, sample_flow=flow)
    inputs = {"flow_voltage": voltage, "inlet_rate": rate}
    yaml_recipe = source / "concentration.yaml"
    from_yaml = run_recipe(yaml_recipe, inputs=inputs, registry=registry)
    columns = [
        "time_utc",
        "number_concentration",
        "quality_flags",
        "calibration_refs",
    ]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        json_recipe = root / "concentration.json"
        json_recipe.write_text(
            json.dumps(yaml.safe_load(yaml_recipe.read_text()))
        )
        run_recipe(
            json_recipe,
            inputs=inputs,
            registry=registry,
            destination=root / "number",
        )
        reopened = open_measurement(root / "number")
        values = reopened.frame.select(columns)
        if not all(
            values.equals(product.frame.select(columns))
            for product in (direct, from_yaml)
        ):
            raise ValueError(
                "Recipe replay differs from the direct calculation"
            )
    report = {
        "relative_s": [
            int((stamp - values["time_utc"][0]).total_seconds())
            for stamp in values["time_utc"]
        ],
        "count_rate": rate.frame["count_rate"].to_list(),
        "sample_flow": flow.frame["sample_flow"].to_list(),
        "flow_relative_s": [
            int((stamp - values["time_utc"][0]).total_seconds())
            for stamp in flow.frame["time_utc"]
        ],
        "concentration": values["number_concentration"].to_list(),
        "flags": values["quality_flags"].to_list(),
        "calibrations": values["calibration_refs"].to_list(),
    }
    print("Time   Number concentration   Calibration   Quality flags")
    for t, concentration, refs, flags in zip(
        report["relative_s"],
        report["concentration"],
        report["calibrations"],
        report["flags"],
        strict=True,
    ):
        value = (
            "missing" if concentration is None else f"{concentration:g} cm⁻³"
        )
        print(
            f"{t} s    {value:10}    {', '.join(refs) or '—':10}   {', '.join(flags) or '—'}"
        )
    return report


# %% [markdown]
# ## Find the true zero and the missing observations
#
# The upper panel plots calibrated flow. The dashed line marks the second
# certificate's effective time. The lower panel shows only concentrations with
# a valid denominator. A row of crosses below the axis identifies counter
# timestamps for which concentration is missing. Read each cross with the
# printed quality flags to distinguish zero flow from an unmatched timestamp.


# %%
def plot_results(report):
    """Compare calibrated flow with the derived number concentration."""
    import matplotlib.pyplot as plt
    from particula import plotting

    seconds = np.asarray(report["relative_s"])
    concentration = np.asarray(report["concentration"], dtype=float)
    valid = np.isfinite(concentration)
    with plt.style.context("particula.default"):
        fig, (flow, number) = plotting.subplots(2, 1, sharex=True, size="wide")
        flow.plot(report["flow_relative_s"], report["sample_flow"], "o")
        flow.set(
            ylabel="Sample flow (L/min)",
            yticks=[0, 0.6, 1.2],
            ylim=(-0.1, 1.45),
        )
        number.plot(seconds[valid], concentration[valid], "o")
        number.plot(
            seconds[~valid],
            np.full(sum(~valid), -2),
            "x",
            color="#983E22",
            markersize=8,
        )
        number.set(
            xlabel="Elapsed time (s)",
            ylabel="Number concentration (cm⁻³)",
            yticks=[-2, 0, 5, 10],
            yticklabels=["Missing", "0", "5", "10"],
            ylim=(-3, 12),
            xticks=seconds,
        )
        for axis in (flow, number):
            axis.axvline(
                2, color="#526463", linestyle="--", linewidth=1, zorder=0
            )
        flow.set_title("Flow calibration changes at 2 s")
        plotting.fit(fig)
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_results(run_example())
    plt.show()

# %% [markdown]
# At seconds 0 and 2, the rate is 100 counts/s and the raw voltage is 2 V.
# The inferred concentration drops from 10 to 5 cm⁻³ because the second
# calibration assigns twice as much sample volume to the same voltage.
# At second 1 the detector reads zero while air still flows, so the recorded
# concentration is zero. At second 3 the denominator is zero, and at second
# 4 the flow record is absent. Neither timestamp supplies a concentration.
#
# All three calculation paths recover the same values and flags, including
# after reopening the JSON result. For a field deployment, a useful next
# test is to compare the two flow predictions with an independent flow
# standard near the certificate change. A counter time series alone cannot
# distinguish a real concentration change from a change in calibrated flow.
