"""Run independent seeds for five-minute, paired collector/blank CUDA pulses."""

from argparse import Namespace

from chamber_gpu import simulate


def main():
    """Run eight prescribed pulse scenarios at each of four random seeds."""
    for case, changes in [
        ("reference", {}),
        ("refined", {"dt": 2.5}),
        ("elevated", {"ion": 100.0}),
        ("single_size", {"ideal": True}),
        ("dispersion", {"vdw": True}),
        ("dispersion_refined", {"vdw": True, "dt": 2.5}),
        ("dispersion_ideal", {"vdw": True, "ideal": True}),
        ("large_parent", {"parent": 1.6}),
    ]:
        for seed in [103, 227, 431, 653]:
            config = dict(
                reps=2048,
                seed=seed,
                dt=5.0,
                duration=300,
                ion=2.0,
                tube=0.1,
                bound="upper",
                parent=1.0,
                ideal=False,
                vdw=False,
                label=f"{case}-{seed}",
            )
            simulate(Namespace(**(config | changes)))


if __name__ == "__main__":
    main()
