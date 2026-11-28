"""Measurement products with unit definitions, grids and source-scoped IDs."""

import polars as pl

from particula.data._binding import Binding


class Measurement:
    """A Polars scan table and its profile, grids and parse diagnostics."""

    def __init__(
        self, scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict
    ):
        """Validate columns and grids against the source layouts."""
        self.lazy_frame = scans.with_columns(
            pl.lit(manifest["source_sha256"]).alias("source_sha256")
        )
        self.lazy_issues = issues
        self.manifest = manifest
        self.grids = manifest["grids"]
        fields = scans.collect_schema()
        layouts = [
            Binding(manifest["profile"], s["header"], s["preamble"])
            for s in manifest["segments"]
        ]
        expected = pl.concat(
            [pl.DataFrame(schema=b.schema).lazy() for b in layouts],
            how="vertical_relaxed",
        ).collect_schema()
        for name, dtype in expected.items():
            if fields.get(name) != dtype:
                raise ValueError(
                    f"Expected measurement column {name} with dtype {dtype}"
                )
        if self.grids != {k: v for b in layouts for k, v in b.grids.items()}:
            raise ValueError("Grid definitions disagree with the source layout")

    @property
    def frame(self) -> pl.DataFrame:
        """Collect all scans in memory; use lazy_frame for disk-sized inputs."""
        return self.lazy_frame.collect()

    @property
    def issues(self) -> pl.DataFrame:
        """Collect source-located diagnostics."""
        return self.lazy_issues.collect()

    def filter(self, predicate):
        """Select scans and retain their source identity and report."""
        return type(self)(
            self.lazy_frame.filter(predicate), self.lazy_issues, self.manifest
        )

    def quality_summary(
        self, field, *, exclude_flags=(), exclude_repaired=False
    ):
        """Summarize a scalar field after explicit source-quality exclusions.

        The mean uses valid samples with equal weight. Report populations,
        the time range affected and the applied policy with the result.
        """
        fields = {f["name"] for f in self.manifest["profile"].get("fields", [])}
        if field not in fields:
            raise ValueError(f"Expected a declared scalar field: {field}")
        flags = (
            [exclude_flags]
            if isinstance(exclude_flags, str)
            else sorted(set(exclude_flags))
        )
        keep = ~pl.col("repaired") if exclude_repaired else pl.lit(True)
        if flags:
            if "vendor_flags" not in self.manifest["profile"]:
                raise ValueError("This source has no vendor status field")
            keep = keep & ~pl.any_horizontal(
                [pl.col("vendor_flags").list.contains(flag) for flag in flags]
            )
        value = pl.col(field).filter(pl.col("_keep"))
        excluded = pl.col("time_utc").filter(~pl.col("_keep"))
        return self.lazy_frame.with_columns(keep.alias("_keep")).select(
            pl.lit(self.manifest["source_sha256"]).alias("source_sha256"),
            pl.lit(flags, dtype=pl.List(pl.String)).alias("excluded_flags"),
            pl.lit(exclude_repaired).alias("exclude_repaired"),
            pl.len().alias("scans"),
            pl.col("_keep").sum().alias("retained"),
            value.count().alias("valid_samples"),
            value.mean().alias("mean"),
            excluded.min().alias("first_excluded"),
            excluded.max().alias("last_excluded"),
        )

    def flag_summary(self):
        """Count each raw vendor condition and report its observed time span."""
        if "vendor_flags" not in self.manifest["profile"]:
            raise ValueError("This source has no vendor status field")
        return (
            self.lazy_frame.select(
                "time_utc", pl.col("vendor_flags").list.unique()
            )
            .with_columns(pl.len().alias("total_scans"))
            .explode("vendor_flags")
            .drop_nulls("vendor_flags")
            .group_by("vendor_flags")
            .agg(
                pl.len().alias("scans"),
                pl.first("total_scans"),
                pl.col("time_utc").min().alias("first_seen"),
                pl.col("time_utc").max().alias("last_seen"),
            )
            .sort("vendor_flags")
        )

    def available_operations(self):
        """Describe compatible operations and any required additional inputs."""
        profile = self.manifest["profile"]
        operations = {"filter": ()}
        if profile.get("fields"):
            operations["quality_summary"] = ("field",)
        if "vendor_flags" in profile:
            operations["flag_summary"] = ()
        return operations


class ParticleCountRate(Measurement):
    """A measurement containing detector count rate in declared units."""

    def __init__(self, scans, issues, manifest):
        """Check the physical product and its source schema."""
        super().__init__(scans, issues, manifest)
        if not any(
            f["quantity"] == "count_rate"
            for f in manifest["profile"].get("fields", [])
        ):
            raise ValueError(
                "ParticleCountRate requires a detector count-rate field"
            )

    def available_operations(self):
        """Report the flow input needed to derive sampled concentration."""
        return {
            **super().available_operations(),
            "number_concentration": ("sample_flow",),
        }


class MobilityNumberSpectrum(Measurement):
    """Number per log10 electrical-mobility diameter on a declared grid."""

    def __init__(self, scans, issues, manifest):
        """Check the physical product and its source schema."""
        super().__init__(scans, issues, manifest)
        spec = manifest["profile"].get("spectrum", {})
        if (
            spec.get("quantity") != "number_per_log10_diameter"
            or spec["axis"]["basis"] != "electrical_mobility"
            or not self.grids
        ):
            raise ValueError(
                "Expected a number-per-decade electrical-mobility spectrum"
            )

    def integrated_number(self) -> pl.LazyFrame:
        """Return lazy scan totals in the declared concentration unit.

        Missing channels leave the total null. Each grid supplies its own
        decimal-log width, derived from the instrument's channels per decade.
        """
        name = self.manifest["profile"]["spectrum"]["name"]
        widths = {k: v["log10_bin_width"] for k, v in self.grids.items()}
        spectrum = pl.col(name).cast(pl.List(pl.Float64))
        total = pl.when(
            spectrum.list.drop_nulls().list.len() == spectrum.list.len()
        ).then(
            spectrum.list.sum()
            * pl.col("grid_id").replace_strict(widths, return_dtype=pl.Float64)
        )
        return self.lazy_frame.select(
            "scan_id", "time_utc", total.alias("integrated_number")
        )

    def available_operations(self):
        """Report the spectrum integral."""
        return {**super().available_operations(), "integrated_number": ()}


def measurement(scans, issues, manifest):
    """Select the product by measured quantities and coordinate basis."""
    profile = manifest["profile"]
    spectrum = profile.get("spectrum", {})
    if (
        spectrum.get("quantity") == "number_per_log10_diameter"
        and spectrum["axis"]["basis"] == "electrical_mobility"
    ):
        return MobilityNumberSpectrum(scans, issues, manifest)
    if any(f["quantity"] == "count_rate" for f in profile.get("fields", [])):
        return ParticleCountRate(scans, issues, manifest)
    return Measurement(scans, issues, manifest)
