"""Example lab extension for converting a flow meter voltage to L/min."""

from particula.data import FieldContract, MeasurementProcess


def linear_flow(signal, *, slope, intercept):
    """Convert volts to L/min with slope in L/min/V and intercept in L/min."""
    return slope * signal + intercept


def flow_process():
    """Describe the lab operation for discovery and recipe validation."""
    return MeasurementProcess(
        "lab.flow.linear",
        "1",
        inputs={"signal": FieldContract("voltage", "V")},
        output=FieldContract("volumetric_flow", "L/min"),
        output_field="sample_flow",
        implementation=linear_flow,
        coefficient_units={"slope": "L/min/V", "intercept": "L/min"},
    )
