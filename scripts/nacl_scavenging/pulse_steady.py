"""Match the steady retrieval to both tracer inputs in the CUDA pulse study."""

from capture_study import rate_models
from instrument import SCAN_CENTERS, save
from other_kernels import chamber_models
from steady import evaluate


def main():
    """Apply the recorded seven channels to identical source specifications."""
    rows = []
    for source in ("reference", "smps_conditioned"):
        for evaluator, keep in [
            (rate_models, {"bare", "dispersion_only", "full"}),
            (chamber_models, {"full_settling_shear_10", "gatti"}),
        ]:
            records = evaluate(
                4.0,
                source,
                analyzer=0.30,
                channel_centers=SCAN_CENTERS,
                condition_on_scan=source == "smps_conditioned",
                pair_models=evaluator,
            )
            rows.extend(
                {
                    k: record[k]
                    for k in (
                        "case",
                        "target_nm",
                        "model",
                        "apparent_cm3_s",
                        "true_capture_cm3_s",
                    )
                }
                for record in records
                if record["model"] in keep
            )
    save("pulse-matched-steady.csv", rows)


if __name__ == "__main__":
    main()
