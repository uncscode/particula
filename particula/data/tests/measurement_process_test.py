"""Public operation contracts tested with independently calculated controls."""

from dataclasses import replace

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("pint")
pytest.importorskip("jsonschema")
from particula.data import (  # noqa: E402
    Calibration,
    Deployment,
    FieldContract,
    MeasurementProcess,
    OperationRegistry,
    describe_profile,
    describe_recipe,
    import_to_parquet,
    open_measurement,
    read_measurement,
    run_recipe,
    save_measurement,
)
from particula.instruments import Instrument, ParticleCounter  # noqa: E402
from particula_docs import documentation_path  # noqa: E402

DATA = documentation_path() / "data/measurement-processing"
UTC_START = "2024-01-01T00:00:00Z"
UTC_STOP = "2024-01-02T00:00:00Z"


def _lab():
    """Register an external callable whose coefficient units are explicit."""
    registry = OperationRegistry()
    process = MeasurementProcess(
        "example.flow.linear",
        "1",
        inputs={"signal": FieldContract("voltage", "V")},
        output=FieldContract("volumetric_flow", "L/min"),
        output_field="flow",
        coefficient_units={"gain": "L/min/V"},
        implementation=lambda signal, gain: signal * gain,
    )
    registry.register(process)
    cert = Calibration(
        "F-1",
        process.operation,
        UTC_START,
        UTC_STOP,
        {"gain": 0.30},
        {"gain": "L/min/V"},
        "bench certificate",
        group="series",
    )
    return registry, process, cert


def _inputs(registry, certs, *, basis="ambient", batch=64):
    meter = Instrument(
        DATA / "flow-profile.yaml",
        registry=registry,
        deployment=Deployment(
            "meter", inlet="A", volume_basis=basis, calibrations=certs
        ),
    )
    counter = ParticleCounter(
        deployment=Deployment("inlet", inlet="A", volume_basis="ambient"),
        registry=registry,
    )
    signal = meter.read(
        DATA / "flow.csv", as_product="voltage", batch_scans=batch
    )
    rate = counter.read(
        DATA / "counter.csv", as_product="rate", batch_scans=batch
    )
    return meter, counter, signal, rate


def test_execute_compose_and_replay(tmp_path):
    """Changing read and output batch sizes preserves exact scan lineage."""
    registry, process, cert = _lab()
    derive = registry.resolve(
        "particula.particle_counter.number_concentration", "1"
    )
    calibrate = process.configure(calibration="series")
    sequence = calibrate.bind(
        signal="voltage", output="calibrated"
    ) | derive.bind(rate="counter", sample_flow="calibrated", output="number")
    assert describe_recipe()["properties"]["recipe_version"]["const"] == "1"
    reference_frame = None
    for batch in (1, 7):
        second = replace(
            cert,
            reference="F-2",
            start="2024-01-01T00:00:02Z",
            coefficients={"gain": 0.60},
        )
        meter, counter, voltage, rate = _inputs(
            registry, (replace(cert, stop=second.start), second), batch=batch
        )
        flow = calibrate.execute(signal=meter)
        direct = derive.execute(rate=counter, sample_flow=flow)
        assert direct.frame["number_concentration"].to_list()[
            :3
        ] == pytest.approx([10, 0, 5])
        aliases = {"voltage": voltage, "counter": rate}
        materialized = sequence.execute(
            inputs=aliases,
            destination=tmp_path / f"measure-{batch}",
            batch_scans=batch,
        )
        assert materialized.product_id == direct.product_id
        assert set(direct.flag_summary().collect()["quality_flags"]) == {
            "unmatched:sample_flow",
            "nonpositive:sample_flow",
        }
        quality = direct.quality_summary(
            "number_concentration", exclude_flags=["unmatched:sample_flow"]
        ).collect()
        assert (
            quality["retained"].item() == 4
            and quality["valid_samples"].item() == 3
        )
        assert materialized.frame.equals(direct.frame)
        assert materialized.frame[
            "parent_sample_flow__signal_scan_id"
        ].to_list() == [2, 3, 4, 5, None]
        assert materialized.frame["calibration_refs"].to_list() == [
            ["F-1"],
            ["F-1"],
            ["F-2"],
            ["F-2"],
            [],
        ]
        parents = direct.manifest["lineage"]["parents"]
        assert parents["rate"]["product_id"] == rate.product_id
        assert (
            parents["sample_flow"]["lineage"]["parents"]["signal"]["product_id"]
            == voltage.product_id
        )
        assert parents["sample_flow"]["product_id"] == flow.product_id
        replay_path = tmp_path / f"measure-{batch}" / "recipe.json"
        replay = run_recipe(replay_path, inputs=aliases, registry=registry)
        assert replay.frame.equals(open_measurement(replay_path.parent).frame)
        assert (
            save_measurement(materialized, replay_path.parent).product_id
            == replay.product_id
        )
        if reference_frame is not None:
            assert reference_frame.equals(materialized.frame)
        reference_frame = materialized.frame


def test_registry_calibration_and_binding_failures():
    """Reject unregistered versions, ambiguous roles and untraceable flow."""
    registry, proc, cert = _lab()
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(proc)
    with pytest.raises(ValueError, match="Unknown"):
        registry.resolve(proc.operation, "2")
    with pytest.raises(ValueError, match="unit"):
        replace(cert, units={}).record()
    for certificates, reason in [
        ((replace(cert, stop="2024-01-01T00:00:02Z"),), "expired"),
        ((cert, replace(cert, reference="overlap")), "Overlapping"),
        ((replace(cert, units={"gain": "kg"}),), "units"),
    ]:
        _, _, voltage, _ = _inputs(registry, certificates)
        with pytest.raises(ValueError, match=reason):
            proc.configure(calibration="series").execute(signal=voltage)
    meter, _, voltage, rate = _inputs(registry, (cert,), basis="standard")
    flow = proc.configure(calibration="F-1").execute(signal=meter)
    derive = registry.resolve(
        "particula.particle_counter.number_concentration", "1"
    )
    with pytest.raises(ValueError, match="volume_basis"):
        derive.execute(rate=rate, sample_flow=flow)
    meter.read(DATA / "flow.csv", as_product="backup")
    with pytest.raises(ValueError, match="voltage.*backup"):
        proc.configure(calibration="F-1").execute(signal=meter)
    assert registry.available_operations(voltage)[proc.operation + "@1"][
        "compatible_roles"
    ] == ["signal"]
    assert proc.operation + "@1" not in registry.available_operations(rate)
    with pytest.raises(ValueError, match="voltage"):
        proc.configure(calibration="F-1").execute(signal=rate)


def test_failed_recipe_publishes_nothing(tmp_path):
    """Preflight errors and lazy execution failures leave no partial directory."""
    registry, process, cert = _lab()
    _, _, voltage, rate = _inputs(registry, (cert,))
    derive = registry.resolve(
        "particula.particle_counter.number_concentration", "1"
    )
    sequence = process.configure(calibration="F-1").bind(
        signal="voltage", output="flow"
    ) | derive.bind(rate="rate", sample_flow="flow", output="number")
    inputs = {"voltage": voltage, "rate": rate}
    target = tmp_path / "published"
    for field, value in [("recipe_version", "9"), ("result", "absent")]:
        with pytest.raises(ValueError):
            run_recipe(
                {**sequence.describe(), field: value},
                inputs=inputs,
                registry=registry,
                destination=target,
            )
    for changes in [
        {"output": "rate"},
        {"inputs": {"rate": "unknown", "sample_flow": "flow"}},
        {"version": "2"},
        {"inputs": {"sample_flow": "flow"}},
    ]:
        config = sequence.describe()
        config["steps"][-1] = {**config["steps"][-1], **changes}
        with pytest.raises(ValueError):
            run_recipe(
                config, inputs=inputs, registry=registry, destination=target
            )
    fail = replace(
        derive,
        operation="example.fail.during_write",
        implementation=lambda rate, sample_flow: (rate * 0 - 1).cast(pl.UInt8),
    )
    with pytest.raises(pl.exceptions.PolarsError):
        (
            process.configure(calibration="F-1").bind(
                signal="voltage", output="flow"
            )
            | fail.bind(rate="rate", sample_flow="flow", output="number")
        ).execute(inputs=inputs, destination=target)
    assert list(tmp_path.iterdir()) == []
    sequence.execute(inputs=inputs, destination=target)
    with pytest.raises(FileExistsError):
        save_measurement(voltage, target)
    assert open_measurement(target).frame["number_concentration"][
        0
    ] == pytest.approx(10)


def test_stateful_filter_across_parts_and_gap(tmp_path):
    """Three-sample causal means retain contributors and reset on a clock gap."""
    profile = DATA / "flow-profile.yaml"
    src = tmp_path / "volts.csv"
    src.write_text("epoch,signal\n0,1\n1,2\n2,3\n3,4\n20,5\n")
    rolling = (
        OperationRegistry()
        .resolve("particula.signal.voltage_mean", "1")
        .configure(window=3, max_gap_seconds=2)
    )
    for batch in (1, 8):
        raw = import_to_parquet(
            src, tmp_path / f"parts-{batch}", profile=profile, batch_scans=batch
        )
        result = rolling.execute(signal=raw).frame
        assert result["mean_voltage"].to_list() == [None, None, 2, 3, None]
        assert result["parent_signal_scan_ids"].to_list() == [
            [2],
            [2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [6],
        ]
        assert result["quality_flags"].to_list()[0] == ["incomplete_window"]


def test_trusted_reader_and_correction_history():
    """User-selected reader code and once-only corrections share the public API."""
    registry, _, _ = _lab()
    registry.register_reader("lab.logger", read_measurement)
    profile = describe_profile(DATA / "flow-profile.yaml")
    custom = Instrument(profile, reader="lab.logger", registry=registry)
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register_reader("lab.logger", read_measurement)
    with pytest.raises(ValueError, match="Unknown reader"):
        Instrument(profile, registry=registry, reader="lab.other")
    op = MeasurementProcess(
        "lab.voltage.offset",
        "1",
        inputs={"signal": FieldContract("voltage", "V")},
        output=FieldContract("voltage", "V"),
        output_field="corrected_voltage",
        implementation=lambda signal: signal + 1,
        correction="offset correction",
    )
    raw = custom.read(DATA / "flow.csv")
    corrected = op.execute(signal=raw)
    assert raw.frame["voltage"].to_list() == [2, 2, 2, 0]
    with pytest.raises(ValueError, match="already applied"):
        op.execute(signal=corrected)


def test_rolling_quality_and_calibrations_follow_window(tmp_path):
    """A missing scan and a certificate change affect overlapping windows."""
    registry, process, first = _lab()
    transition = "2024-01-01T00:00:02Z"
    second = replace(first, start=transition, reference="F-2")
    meter, _, _, _ = _inputs(
        registry, (replace(first, stop=transition), second)
    )
    src = tmp_path / "volts.csv"
    src.write_text(
        "epoch,signal\n1704067200,1\n1704067201,.\n1704067202,3\n"
        "1704067203,4\n1704067204,5\n1704067220,6\n"
    )
    rolling = registry.resolve("particula.signal.voltage_mean", "1")
    calibrated = replace(
        process,
        mode="ordered",
        implementation=lambda signal, gain, **params: rolling.implementation(
            signal * gain, **params
        ),
        parameter_schema=rolling.parameter_schema,
        history_parameters=rolling.history_parameters,
    ).configure(window=3, max_gap_seconds=2, calibration="series")
    result = calibrated.execute(signal=meter.read(src)).frame
    assert result["flow"].to_list() == pytest.approx(
        [None, None, None, None, 1.2, None]
    )
    assert result["quality_flags"].to_list() == [
        ["incomplete_window"],
        ["incomplete_window", "missing:signal"],
        ["missing:signal"],
        ["missing:signal"],
        [],
        ["incomplete_window"],
    ]
    assert result["calibration_refs"].to_list() == [
        ["F-1"],
        ["F-1"],
        ["F-1", "F-2"],
        ["F-1", "F-2"],
        ["F-2"],
        ["F-2"],
    ]


def test_whole_record_propagates_each_contributors_quality(tmp_path):
    """A warning anywhere in the baseline affects every centered sample."""
    registry, process, first = _lab()
    transition = "2024-01-01T00:00:02Z"
    second = replace(first, start=transition, reference="F-2")
    profile = describe_profile(DATA / "flow-profile.yaml")
    profile["input"]["record_width"] = 3
    profile["vendor_flags"] = {"column": "status", "separator": ";"}
    meter = Instrument(
        profile,
        registry=registry,
        deployment=Deployment(
            "meter", calibrations=(replace(first, stop=transition), second)
        ),
    )
    src = tmp_path / "volts.csv"
    src.write_text(
        "epoch,signal,status\n1704067200,1,\n1704067201,2,drift\n1704067202,3,\n"
    )
    raw = meter.read(src)
    centered = replace(
        process,
        mode="whole_record",
        implementation=lambda signal, gain: signal * gain
        - (signal * gain).mean(),
    ).configure(calibration="series")
    result = centered.execute(signal=raw)
    assert result.frame["flow"].to_list() == pytest.approx([-0.3, 0, 0.3])
    assert result.frame["quality_flags"].to_list() == [["drift"]] * 3
    assert result.frame["calibration_refs"].to_list() == [["F-1", "F-2"]] * 3
    assert (
        result.quality_summary("flow", exclude_flags=["drift"])
        .collect()["retained"]
        .item()
        == 0
    )
    assert centered.execute(signal=raw.filter(pl.lit(False))).frame.is_empty()


def test_array_roles_compare_selected_fields_coordinates(tmp_path):
    """Equal shapes require matching axis units and coordinate values."""
    profile = describe_profile("cpc-3010")
    profile["on_bad_record"] = "error"
    field = profile.pop("fields")[1]
    field.pop("column")
    axis = {"name": "x", "unit": "m", "values": [0, 1]}
    left = dict(field, name="left", columns=[1, 2], axes=[axis])
    right = dict(field, name="right", columns=[3, 4], axes=[axis])
    profile["arrays"] = [left, right]
    src = tmp_path / "pair.csv"
    src.write_text("1700000040,1,2,4,8\n")
    operation = MeasurementProcess(
        "lab.pair.difference",
        "1",
        inputs={
            side: FieldContract("temperature", "K", field=side, axes=("x",))
            for side in ("left", "right")
        },
        output=FieldContract("temperature", "K"),
        output_field="difference",
        implementation=lambda left, right: right.arr.get(1) - left.arr.get(1),
    )
    original = read_measurement(src, profile=profile)
    assert (
        operation.execute(left=original, right=original)
        .frame["difference"]
        .item()
        == 6
    )
    for changed_axis in [dict(axis, values=[0, 2]), dict(axis, unit="cm")]:
        right["axes"] = [changed_axis]
        changed = read_measurement(src, profile=profile)
        with pytest.raises(ValueError, match="common coordinate grid"):
            operation.execute(left=changed, right=changed)
        # An unrelated variable's axis does not change the selected left grid.
        assert (
            operation.execute(left=changed, right=original)
            .frame["difference"]
            .item()
            == 6
        )


def test_signed_clock_correction_retains_reported_time(tmp_path):
    """Offsets apply once on reported time, without changing record identity."""
    from particula.data import correct_clock

    src = read_measurement(DATA / "counter.csv", profile="cpc-3010")
    correction = dict(
        start=UTC_START,
        stop="2024-01-01T00:00:02Z",
        offset_seconds=-0.25,
        source="reference pulse A",
    )
    corrected = correct_clock(src, **correction)
    assert corrected.product_id != src.product_id
    assert corrected.frame["scan_id"].equals(src.frame["scan_id"])
    assert corrected.frame["reported_time_utc"].equals(src.frame["time_utc"])
    shifts = (
        corrected.frame["time_utc"] - src.frame["time_utc"]
    ).dt.total_milliseconds()
    assert shifts.to_list() == [-250, -250, 0, 0, 0]
    with pytest.raises(ValueError, match="Overlapping"):
        correct_clock(corrected, **correction)
    reopened = save_measurement(corrected, tmp_path / "corrected")
    assert reopened.frame.equals(corrected.frame)
    assert (
        reopened.manifest["clock_corrections"][0]["source"]
        == "reference pulse A"
    )


def test_registered_batch_strategy_uses_both_sinks(tmp_path):
    """Construct fresh strategy instances for collection and streaming import."""
    from particula.data import MeasurementReader

    class LabReader(MeasurementReader):
        """Instrument-specific decoder with the standard batch interface."""

        def batches(self):
            """Record the decoder revision in the completed manifest."""
            yield from super().batches()
            self.manifest["parser_version"] = "lab-1"

    registry = OperationRegistry()
    registry.register_batch_reader("lab.batch", LabReader)
    source = DATA / "flow.csv"
    profile = DATA / "flow-profile.yaml"
    raw = Instrument(profile, reader="lab.batch", registry=registry).read(
        source
    )
    disk = import_to_parquet(
        source,
        tmp_path / "lab",
        profile=profile,
        reader_factory=registry.batch_reader("lab.batch"),
        batch_scans=1,
    )
    assert raw.frame.equals(disk.frame)
    assert raw.product_id == disk.product_id
    assert disk.manifest["parser_version"] == "lab-1"


def test_axis_contract_and_vendor_correction():
    """A scalar role cannot silently consume an array on the wrong basis."""
    spectrum = read_measurement(
        documentation_path() / "data/measurement-import/mobility.csv",
        profile="smps-aim",
    )
    channel = MeasurementProcess(
        "lab.spectrum.first_channel",
        "1",
        inputs={
            "spectrum": FieldContract(
                "number_per_log10_diameter",
                "1/m^3",
                axes=("diameter",),
                basis="electrical_mobility",
            )
        },
        output=FieldContract("number_per_log10_diameter", "1/m^3"),
        output_field="first_channel",
        implementation=lambda spectrum: spectrum.arr.get(0),
    )
    result = channel.execute(spectrum=spectrum)
    assert result.frame["first_channel"][0] == pytest.approx(
        spectrum.frame["number_spectrum"][0][0]
    )
    with pytest.raises(ValueError, match="already applied"):
        replace(channel, correction="Multiple Charge Correction").execute(
            spectrum=spectrum
        )
    with pytest.raises(ValueError, match="compatible fields"):
        replace(
            channel,
            inputs={
                "spectrum": replace(
                    channel.inputs["spectrum"], basis="aerodynamic"
                )
            },
        ).execute(spectrum=spectrum)


def test_swap_compatible_flow_source_and_preflight(tmp_path):
    """A replacement meter changes concentration and its traceable parent ID."""
    from particula.data import validate_recipe

    registry, process, cert = _lab()
    _, _, voltage, rate = _inputs(registry, (cert,))
    file = tmp_path / "second.csv"
    file.write_text("epoch,signal\n1704067200,4\n")
    replacement = Instrument(
        DATA / "flow-profile.yaml",
        deployment=Deployment(
            "another meter",
            inlet="A",
            volume_basis="ambient",
            calibrations=(cert,),
        ),
    ).read(file)
    derive = registry.resolve(
        "particula.particle_counter.number_concentration", "1"
    )
    recipe = (
        process.configure(calibration="F-1").bind(signal="volts", output="flow")
        | derive.bind(rate="rate", sample_flow="flow", output="ambient")
    ).describe()
    for src, expected in [(voltage, 10), (replacement, 5)]:
        inputs = {"volts": src, "rate": rate}
        assert (
            validate_recipe(recipe, inputs=inputs, registry=registry) == recipe
        )
        result = run_recipe(recipe, inputs=inputs, registry=registry)
        assert result.frame["number_concentration"][0] == pytest.approx(
            expected
        )
        parent = result.manifest["lineage"]["parents"]["sample_flow"][
            "lineage"
        ]["parents"]["signal"]
        assert parent["product_id"] == src.product_id
    with pytest.raises(ValueError, match="Incompatible unit"):
        FieldContract("voltage", "kg")
