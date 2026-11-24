"""Particula builders for spatial flow and recorded forcing providers."""

from particula.abc_builder import BuilderABC
from particula.abc_factory import StrategyFactoryABC
from particula.coupling.flow_replay import GridReplay, TrajectoryReplay
from particula.coupling.flows import DuneFlowStrategy, SoundingFlow, UniformFlow
from particula.coupling.prescribed import DuneFlow, Sounding
from particula.dynamics.parcel_trace import ParcelTrace
from particula.util.convert_units import get_unit_conversion


def _convert(units, target, *, value):
    return (
        value
        if units == target
        else get_unit_conversion(units, target, value=value)
    )


class UniformFlowBuilder(BuilderABC):
    """Build a spatially uniform flow with conversion to SI."""

    def __init__(self):
        """Declare the parameters required by this provider."""
        super().__init__(["velocity", "temperature", "pressure", "vapor"])
        self.velocity = self.temperature = self.pressure = self.vapor = None

    def set_velocity(self, value, units="m/s"):
        """Set the east, north and vertical wind components."""
        self.velocity = tuple(_convert(units, "m/s", value=v) for v in value)
        return self

    def set_temperature(self, value, units="K"):
        """Set the air temperature."""
        self.temperature = _convert(units, "K", value=value)
        return self

    def set_pressure(self, value, units="Pa"):
        """Set the total pressure."""
        self.pressure = _convert(units, "Pa", value=value)
        return self

    def set_vapor(self, value, units="kg/kg"):
        """Set water mass per kg dry air."""
        self.vapor = _convert(units, "kg/kg", value=value)
        return self

    def build(self):
        """Construct validated constant fields."""
        self.pre_build_check()
        return UniformFlow(
            self.velocity, self.temperature, self.pressure, self.vapor
        )


class SoundingFlowBuilder(BuilderABC):
    """Build from a normalized SI sounding object or CSV path."""

    def __init__(self):
        """Declare the parameters required by this provider."""
        super().__init__(["sounding"])
        self.sounding = None

    def set_sounding(self, value):
        """Supply a sounding or normalized CSV."""
        self.sounding = (
            value if isinstance(value, Sounding) else Sounding.from_csv(value)
        )
        return self

    def build(self):
        """Construct horizontally uniform profile forcing."""
        self.pre_build_check()
        return SoundingFlow(self.sounding)


class DuneFlowBuilder(SoundingFlowBuilder):
    """Build the Gaussian dune with a vertical sounding."""

    def __init__(self):
        """Declare the parameters required by this provider."""
        super().__init__()
        self.required_parameters += ["height", "width", "lid"]
        self.height = self.width = self.lid = None

    def set_height(self, value, units="m"):
        """Set crest elevation."""
        self.height = _convert(units, "m", value=value)
        return self

    def set_width(self, value, units="m"):
        """Set the Gaussian standard deviation."""
        self.width = _convert(units, "m", value=value)
        return self

    def set_lid(self, value, units="m"):
        """Set lid height above datum."""
        self.lid = _convert(units, "m", value=value)
        return self

    def build(self):
        """Construct terrain-following forcing."""
        self.pre_build_check()
        return DuneFlowStrategy(
            DuneFlow(self.sounding, self.height, self.width, self.lid)
        )


class TrajectoryReplayBuilder(BuilderABC):
    """Build from a native parcel CSV and its sibling metadata."""

    def __init__(self):
        """Declare the parameters required by this provider."""
        super().__init__(["path"])
        self.path = None

    def set_path(self, value):
        """Select the forcing archive."""
        self.path = value
        return self

    def build(self):
        """Load a trajectory by its recorded identity."""
        self.pre_build_check()
        return TrajectoryReplay(ParcelTrace.load(self.path))


class GridReplayBuilder(TrajectoryReplayBuilder):
    """Load accepted grid forcing from its numerical archive."""

    def build(self):
        """Validate and return exact gridded forcing."""
        self.pre_build_check()
        return GridReplay.load(self.path)


class FlowFactory(StrategyFactoryABC):
    """Construct an independent builder for each supported flow provider."""

    def get_builders(self):
        """Map flow names to fresh builders."""
        return {
            "uniform": UniformFlowBuilder(),
            "sounding": SoundingFlowBuilder(),
            "dune": DuneFlowBuilder(),
            "trajectory_replay": TrajectoryReplayBuilder(),
            "grid_replay": GridReplayBuilder(),
        }
