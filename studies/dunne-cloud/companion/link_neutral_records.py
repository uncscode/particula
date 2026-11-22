"""Make an outcome-blind condition crosswalk. Original archives remain intact."""

import csv
import hashlib
import io
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
GROUPS = (208, 223, 248, 278, 292)


def load_chamber_figures():
    path = HERE / "kuerten2019-data.zip"
    if hashlib.md5(path.read_bytes()).hexdigest() != "1d96e9534be574b2b062ce264a4c981f":
        raise ValueError("Zenodo archive checksum mismatch")
    with ZipFile(path) as z:
        def read_table(name):
            return list(csv.DictReader(io.StringIO(z.read(name).decode()), delimiter="\t"))

        f3 = read_table("Figure_03.txt")
        f4 = read_table("Figure_04_CLOUD_data.txt")
    observations = {}
    for nominal in GROUPS:
        prefix = f"T{nominal}K_"
        for line, row in enumerate(f4, start=2):
            if not row[prefix + "J17"]:
                continue
            values = {k: float(row[prefix + k]) for k in
                      ("Temp", "SA", "NH3", "J17", "J17_error_minus", "J17_error_plus")}
            matches = [(i, r) for i, r in enumerate(f3, start=2)
                       if float(r["temperature"]) == values["Temp"]
                       and float(r["J_from_CLOUD"]) == values["J17"]]
            values["fig3_lines"] = "|".join(str(i) for i, _ in matches)
            values["santiago_J"] = float(matches[0][1]["J_from_model"]) if len(matches) == 1 else ""
            values["fig4_group_K"] = nominal
            values["fig4_txt_line"] = line
            observations[f"{nominal}K-L{line}"] = values
    return observations


def load_dunne():
    with (HERE.parent / "raw" / "cloud_data_table.csv").open() as stream:
        rows = csv.reader(stream)
        next(rows)
        records = {}
        for line, row in enumerate(rows, start=2):
            temperature, acid, ammonia, q, rh, rate = map(float, row)
            if q == 0:
                records[line] = {"Temp": temperature, "SA": acid * 1e6, "NH3": ammonia,
                                 "RH_percent": rh, "J17": rate}
    return records


def candidates(dunne, observations, acid_ratio):
    pairs = {}
    for line, original in dunne.items():
        nearby = []
        for tag, comparison in observations.items():
            acid, nh3 = original["SA"], original["NH3"]
            if abs(original["Temp"] - comparison["Temp"]) > 0.055:
                continue
            if max(acid, comparison["SA"]) / min(acid, comparison["SA"]) > acid_ratio:
                continue
            if (abs(nh3 - comparison["NH3"]) > .005 and
                max(nh3, comparison["NH3"]) / min(nh3, comparison["NH3"]) > 1.25):
                continue
            nearby.append(tag)
        pairs[line] = nearby
    return pairs


def write_csv(name, records):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def main():
    original, other = load_dunne(), load_chamber_figures()
    exact = [(i, tag) for i, a in original.items() for tag, b in other.items()
             if all(math.isclose(a[k], b[k], rel_tol=1e-14, abs_tol=1e-14)
                    for k in ("Temp", "SA", "NH3", "J17"))]
    print("Exact neutral numerical matches on T, acid, ammonia, J:", len(exact))
    for scale in (1.02, 1.05, 1.10):
        pairs = candidates(original, other, scale)
        n_reverse = Counter(tag for tags in pairs.values() for tag in tags)
        records = []
        links = []
        for line, tags in pairs.items():
            a = original[line]
            state = "no_candidate"
            if tags:
                state = "condition_unique" if len(tags) == 1 and n_reverse[tags[0]] == 1 else "ambiguous"
            records.append({"dunne_csv_line": line, **a, "status": state,
                            "n_candidates": len(tags), "fig4_candidates": "|".join(tags)})
            for tag in tags:
                b = other[tag]
                lower, upper = b["J17"] - b["J17_error_minus"], b["J17"] + b["J17_error_plus"]
                links.append({"dunne_csv_line": line, "kuerten_fig4_id": tag,
                              "mutually_unique_condition": state == "condition_unique",
                              "dunne_candidates": len(tags), "kuerten_candidates": n_reverse[tag],
                              "dunne_J": a["J17"], "kuerten_J": b["J17"],
                              "kuerten_J_lower": lower, "kuerten_J_upper": upper,
                              "dunne_J_inside_kuerten_bars": lower <= a["J17"] <= upper,
                              "kuerten_fig3_lines": b["fig3_lines"], "santiago_J": b["santiago_J"]})
        print("acid max/min", scale, Counter(r["status"] for r in records),
              "cross-archive candidate edges",len(links))
        if scale == 1.05:
            write_csv("dunne_neutral_crosswalk.csv", records)
            write_csv("dunne_kuerten_candidate_edges.csv", links)
            write_csv("kuerten_neutral_observations.csv",
                      [{"kuerten_fig4_id": tag, **b} for tag, b in other.items()])
            print("Mutually unique candidates with Dunne J inside later Fig 4 bars:",
                  Counter(r["dunne_J_inside_kuerten_bars"] for r in links if r["mutually_unique_condition"]))


if __name__ == "__main__":
    main()
