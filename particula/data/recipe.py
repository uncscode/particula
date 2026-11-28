"""Validate and replay measurement processes from Python, YAML or JSON."""

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike

from jsonschema import Draft202012Validator

from particula.data.measurement import Measurement
from particula.data.process import MeasurementProcess, _unit_transform
from particula.data.product import DerivedMeasurement, product_fields
from particula.data.profile import ProfileSource, _decode, _resource, canonical
from particula.data.registry import OperationRegistry
from particula.data.snapshot import save_measurement


def describe_recipe() -> dict:
    """Return the installed processing recipe schema for offline validation."""
    return json.loads(_resource("recipe.schema.json"))


@dataclass(frozen=True)
class Recipe:
    """Ordered, explicitly bound operations for Python composition."""

    steps: tuple[dict, ...]
    processes: tuple[MeasurementProcess, ...]

    @classmethod
    def from_process(
        cls, process: MeasurementProcess, inputs: dict[str, str], output: str
    ) -> "Recipe":
        """Create a single step with named input and output aliases."""
        return cls(
            (
                {
                    "operation": process.operation,
                    "version": process.version,
                    "parameters": process.parameters,
                    "inputs": inputs,
                    "output": output,
                },
            ),
            (process,),
        )

    def __or__(self, other: "Recipe") -> "Recipe":
        """Compose ordered steps with explicit input roles."""
        return Recipe(
            self.steps + other.steps, self.processes + other.processes
        )

    def describe(self) -> dict:
        """Return an independently editable prescription for JSON or YAML."""
        return {
            "recipe_version": "1",
            "steps": deepcopy(list(self.steps)),
            "result": self.steps[-1]["output"],
        }

    def execute(
        self,
        *,
        inputs: Mapping[str, Measurement],
        destination: str | PathLike[str] | None = None,
        batch_scans: int = 65536,
    ) -> DerivedMeasurement:
        """Validate and execute the sequence and optionally publish it."""
        return _execute(
            self.describe(), self.processes, inputs, destination, batch_scans
        )


def _check_structure(recipe: dict) -> None:
    error = next(
        Draft202012Validator(describe_recipe()).iter_errors(recipe), None
    )
    if error:
        raise ValueError(
            f"Invalid recipe at {list(error.absolute_path)}: {error.message}"
        )
    if recipe["steps"][-1]["output"] != recipe["result"]:
        raise ValueError("Recipe result must be the final step output")


def _check_bindings(
    recipe: dict,
    processes: tuple[MeasurementProcess, ...],
    inputs: Mapping[str, Measurement],
) -> None:
    """Check aliases, units and parameters before calling an operation."""
    fields = {name: product_fields(product) for name, product in inputs.items()}
    for process, step in zip(processes, recipe["steps"], strict=True):
        roles = step["inputs"]
        if set(roles) != set(process.inputs):
            raise ValueError(
                f"Expected {list(process.inputs)} for {process.operation}"
            )
        if step["output"] in fields:
            raise ValueError(f"Output alias already exists: {step['output']}")
        for role, alias in roles.items():
            if alias not in fields:
                raise ValueError(
                    f"Unresolved or forward input {alias!r} for {role!r}"
                )
            contract = process.inputs[role]
            definition = contract.resolve(fields[alias])
            _unit_transform(definition["output_unit"], contract.unit)
        process._arguments()
        fields[step["output"]] = [
            {
                "name": process.output_field,
                "quantity": process.output.quantity,
                "output_unit": process.output.unit,
            }
        ]


def _execute(
    recipe: dict,
    processes: tuple[MeasurementProcess, ...],
    inputs: Mapping[str, Measurement],
    destination: str | PathLike[str] | None,
    batch_scans: int,
) -> DerivedMeasurement:
    _check_structure(recipe)
    _check_bindings(recipe, processes, inputs)
    products = dict(inputs)
    for process, step in zip(processes, recipe["steps"], strict=True):
        result = process.execute(
            **{role: products[alias] for role, alias in step["inputs"].items()}
        )
        products[step["output"]] = result
    result = DerivedMeasurement(
        result.lazy_frame,
        result.lazy_issues,
        {**result.manifest, "recipe": json.loads(canonical(recipe))},
    )
    if destination is not None:
        reopened = save_measurement(
            result, destination, batch_scans=batch_scans
        )
        if not isinstance(reopened, DerivedMeasurement):
            raise TypeError("Saved recipe result lost its product type")
        return reopened
    return result


def _resolve(
    recipe: ProfileSource, registry: OperationRegistry
) -> tuple[dict, tuple[MeasurementProcess, ...]]:
    definition = _decode(recipe)
    _check_structure(definition)
    processes = tuple(
        registry.resolve(step["operation"], step["version"]).configure(
            **step.get("parameters", {})
        )
        for step in definition["steps"]
    )
    return definition, processes


def validate_recipe(
    recipe: ProfileSource,
    *,
    inputs: Mapping[str, Measurement],
    registry: OperationRegistry,
) -> dict:
    """Validate a prescription and its named inputs before executing any step.

    Args:
        recipe: Recipe mapping or path to YAML or JSON.
        inputs: Named measurements available to the first or later steps.
        registry: Trusted operations used for signature and parameter checks.

    Returns:
        Decoded prescription with validated operation versions, parameters,
        field units and upstream aliases. Row-level checks occur at execution.

    Raises:
        ValueError: An operation, physical role, unit or output is unresolved.
    """
    definition, processes = _resolve(recipe, registry)
    _check_bindings(definition, processes, inputs)
    return definition


def run_recipe(
    recipe: ProfileSource,
    *,
    inputs: Mapping[str, Measurement],
    registry: OperationRegistry,
    destination: str | PathLike[str] | None = None,
    batch_scans: int = 65536,
) -> DerivedMeasurement:
    """Replay a recipe with explicitly supplied code and inputs.

    Args:
        recipe: Recipe mapping or path to JSON or YAML.
        inputs: Named input measurements, including collection members.
        registry: Trusted Python implementations for the referenced versions.
        destination: Optional path to atomically publish the completed result.
        batch_scans: Maximum rows per Parquet row group when publishing.

    Returns:
        Derived measurement with parent lineage and the replayable recipe.

    Raises:
        ValueError: Recipe structure, input bindings or physical contracts fail.
    """
    definition, processes = _resolve(recipe, registry)
    return _execute(definition, processes, inputs, destination, batch_scans)
