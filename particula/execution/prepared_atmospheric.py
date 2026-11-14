"""Compile explicit ion collision laws into the existing prepared sampler."""

from dataclasses import replace


def plan_atmospheric(process, _aerosol):
    """Freeze the complete physical collision configuration for restart."""
    from particula.execution.prepared_processes import _ProcessPlan

    return _ProcessPlan(
        "coagulation",
        (("atmospheric", True),)
        + tuple(
            (name, getattr(process, name))
            for name in (
                "positive_species",
                "negative_species",
                "charged_model",
                "omitted",
                "recombination",
                "attachment_model",
                "positive_mobility",
                "negative_mobility",
                "ion_radius",
            )
        ),
    )


def configure_atmospheric(plan, owner, prepared):
    """Retain derived classification and rate scratch; no checkpoint payload."""
    from particula.dynamics.ions.attachment_table import (
        centered_attachment_tables,
    )
    from particula.dynamics.ions.coagulation import (
        NEGATIVE_ION_MASS,
        POSITIVE_ION_MASS,
    )
    from particula.gpu.dynamics.ion_pair_data import IonPairData
    from particula.gpu.kernels.ion_coagulation import (
        cache_attachment_rates,
        cache_mobility_rates,
        classify_ions,
    )

    wp = owner._wp
    work = prepared.brownian_parallel
    if work is None:
        raise ValueError("atmospheric coagulation requires charged pair work")
    args, device = work.selector_inputs, work.device
    shape = args[1].shape
    kinds = wp.empty(shape, dtype=wp.int32, device=device)
    rates = wp.empty((*shape, 2), dtype=wp.float64, device=device)
    presence = wp.empty((shape[0], 2), dtype=wp.int32, device=device)
    tables = wp.array(
        centered_attachment_tables(), dtype=wp.float64, device=device
    )
    data = IonPairData()
    data.enabled = 1
    data.kinds = kinds
    data.attachment = rates
    data.recombination = wp.float64(plan.values["recombination"])
    values = plan.values
    classify_args = (
        args[0],
        args[1],
        args[23],
        args[4],
        args[5],
        values["positive_species"],
        values["negative_species"],
        wp.float64(POSITIVE_ION_MASS),
        wp.float64(NEGATIVE_ION_MASS),
        kinds,
        presence,
        owner._device_failure,
        int(values.get("attachment_model", "lf2013") == "mobility"),
    )
    rate_args = (
        kinds,
        presence,
        args[15],
        args[23],
        ("reject", "lower", "upper").index(values["omitted"]),
        tables,
        rates,
        owner._device_failure,
    )

    rate_kernel = cache_attachment_rates
    if values.get("attachment_model", "lf2013") == "mobility":
        rate_kernel = cache_mobility_rates
        rate_args = (
            kinds,
            presence,
            args[15],
            args[23],
            args[0],
            args[4],
            args[5],
            wp.float64(POSITIVE_ION_MASS),
            wp.float64(NEGATIVE_ION_MASS),
            wp.float64(values["positive_mobility"]),
            wp.float64(values["negative_mobility"]),
            wp.float64(values["ion_radius"]),
            rates,
            owner._device_failure,
        )

    def setup():
        wp.launch(
            classify_ions, dim=shape[0], inputs=classify_args, device=device
        )
        wp.launch(
            rate_kernel,
            dim=rates.shape,
            inputs=rate_args,
            device=device,
        )

    return replace(
        prepared,
        brownian_parallel=replace(work, ion_data=data, ion_setup=setup),
    )


def validate_atmospheric_companions(plans):
    """Reject conflicting source/cleanup/exclusion material identities."""
    from particula.dynamics.ions.coagulation import (
        NEGATIVE_ION_MASS,
        POSITIVE_ION_MASS,
    )

    atmospheric = next((p for p in plans if p.values.get("atmospheric")), None)
    if atmospheric is None:
        return
    values = atmospheric.values
    for plan in plans:
        if plan.family in ("ion_source", "ion_cleanup", "ion_condensation"):
            if any(
                plan.values[key] != values[key]
                for key in ("positive_species", "negative_species")
            ):
                raise ValueError("ion processes require matching material maps")
        if plan.family == "ion_source" and (
            plan.values["positive_mass"] != POSITIVE_ION_MASS
            or plan.values["negative_mass"] != NEGATIVE_ION_MASS
        ):
            raise ValueError("LF2013 source requires 150 u and 90 u ions")
        if plan.family == "condensation" and not any(
            p.family == "ion_condensation" for p in plans
        ):
            raise ValueError(
                "atmospheric condensation requires IonCondensationExclusion"
            )
