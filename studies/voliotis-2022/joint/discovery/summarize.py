"""Build the twenty-round audit table, literature crosswalk and study figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results/discovery"


def summarize_search():
    """Publish fit chronology, molecular analogues and per-run comparison."""
    records = [
        json.loads(f.read_text()) for f in sorted(OUT.glob("round_*.json"))
    ]
    controls = pd.DataFrame(
        json.loads((OUT / "paired_controls.json").read_text())
    ).set_index(["round", "objective"])
    prefix = {
        (r["size"], r["objective"]): r["fit"]["score"]
        for r in json.loads((OUT / "prefix_fits.json").read_text())
    }
    mechanism = json.loads((OUT / "mechanism.json").read_text())
    reactions = {r["key"]: r for r in mechanism["reactions"]}
    rows = []
    paragraphs = []
    for r in records:
        n = r["round"]
        key = r["key"]
        prop = json.loads((OUT / f"proposal_{n:02}.json").read_text())
        lit = json.loads((OUT / f"decision_{n:02}.json").read_text())
        if not r["accepted"]:
            paragraphs.append(
                f"### Attempt {n} · {prop['equation']}\n\n"
                f"{lit['difference']} [{lit['citation']}]({lit['url']})"
            )
            continue
        rxn = reactions[key]
        units = "h⁻¹" if rxn["b"] < 0 else "m³ air µmol⁻¹ h⁻¹"
        for obj in ("mass", "joint"):
            s = r["fits"][obj]
            null = controls.loc[(n, obj), "null_score"]
            rows.append(
                dict(
                    round=n,
                    key=key,
                    objective=obj,
                    equation=prop["equation"],
                    literature_match=lit["match"],
                    before=prop["prior"][obj]["score"],
                    shadow=prop["null"][obj]["score"],
                    proposed=prop["initial_fit"][obj]["score"],
                    retained=s["score"],
                    retrospective=prefix[(n, obj)],
                    equal_extra_refit_null=null,
                    gain_over_null=null - s["score"],
                    entry_rate=s["k"][r["index"]],
                    final_rate=rxn[f"{obj}_fit_rate"],
                    units=units,
                )
            )
        paragraphs.append(
            f"### {n}. {prop['equation']}\n\n{lit['analogue']} "
            f"{lit['evidence']} [{lit['citation']}]({lit['url']})\n\n"
            f"**Scope of the match.** {lit['difference']}\n\n"
            f"The final effective rate was {rxn['mass_fit_rate']:.5g} {units} "
            f"in the mass fit and {rxn['joint_fit_rate']:.5g} {units} "
            "in the joint fit."
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rounds.csv", index=False)
    (OUT / "REACTION_EVIDENCE.md").write_text(
        "# Twenty fitted chamber reactions and their closest "
        "published analogues\n\n"
        "A, C and I indicate α-pinene, o-cresol and isoprene. G "
        "denotes the gas intermediate from the original delay "
        "model, V the secondary oxygenated-vapor reservoir, P the "
        "labile suspended organic product, S a stabilized suspended "
        "product and F terminal volatile products. The rate "
        "multiplies the effective pool concentration in the model. "
        "A match to a reaction family supports chemical "
        "plausibility. Each entry records which molecular "
        "identities, rates or conditions the cited paper "
        "establishes.\n\n" + "\n\n".join(paragraphs) + "\n"
    )


def plot():
    """Draw the comparison from the published scores."""
    table = pd.read_csv(OUT / "rounds.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    base = json.loads((OUT / "baseline.json").read_text())
    score = pd.read_csv(OUT / "final_scores.csv")
    for obj, color, label in [
        ("mass", "#a34726", "Mass fit · fixed wall"),
        ("joint", "#17677a", "Mass and counts · fitted wall"),
    ]:
        subset = table[table.objective == obj]
        ax.plot(
            np.arange(len(subset) + 1),
            100 * np.r_[base[obj]["score"], subset.retained],
            ".-",
            color=color,
            label=label,
        )
        v = score[
            (score.channels == 20) & (score.objective == obj)
        ].score.item()
        ax.plot(
            np.arange(1, len(subset) + 1),
            100 * subset.retrospective,
            color=color,
            linestyle="--",
            lw=1.3,
        )
        ax.scatter(21, v * 100, marker="D", color=color, s=27)
    ax.set(
        xticks=[0, 5, 10, 15, 20],
        xlabel="Literature-matched reactions admitted",
        ylabel="Worst trajectory error (%)",
        title="What did each added reaction buy?",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.02,
        0.04,
        "Dashed: retrospective refits · diamonds: final fits",
        fontsize=8,
        transform=ax.transAxes,
    )
    data = (
        pd.read_csv(OUT / "per_run.csv")
        .query("objective=='mass'")
        .pivot(index="run", columns="channels", values="mass")
    )
    idx = np.arange(len(data))
    bx.vlines(idx, 100 * data[0], 100 * data[20], color="#a1a6a6", lw=1.5)
    bx.scatter(
        idx,
        100 * data[0],
        label="No new reactions",
        facecolors="white",
        edgecolors="#525757",
        s=28,
    )
    bx.scatter(idx, 100 * data[20], label="20 reactions", color="#a34726", s=22)
    bx.axhline(10, ls=":", lw=1, color="#647077")
    bx.set(
        xticks=idx,
        xticklabels=data.index,
        xlabel="Voliotis experiment",
        ylabel="Normalized SOA mass error (%)",
        title="Which experiments changed?",
    )
    bx.legend(frameon=False, fontsize=8)
    for axis in (ax, bx):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.17)
    fig.savefig(OUT / "reaction-discovery.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-checkpoints", action="store_true")
    if parser.parse_args().from_checkpoints:
        summarize_search()
    plot()
