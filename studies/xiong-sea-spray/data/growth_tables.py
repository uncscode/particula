"""Recover Xiong 2026 growth-factor tables and reported 100 nm changes."""

import csv
import re
import subprocess


def save(path, fields, records):
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(fields.split(","))
        writer.writerows(records)


def table_s1(supp, output):
    full = subprocess.check_output(["pdftotext", "-layout", supp, "-"]).decode()
    table = full.split("GF(3.5h)", 1)[1].split("Table S2.")[0]
    rows = []
    for line in table.splitlines():
        values = re.findall(r"(\d[.]\d+)\s*±\s*(\d[.]\d+)", line)
        if not values:
            continue
        assert len(values) == 7
        prefix = line[: line.index(values[0][0])].split()
        if len(prefix) == 2:
            diameter = int(prefix[0])
        rh = int(prefix[-1])
        for j, (gf, sd) in enumerate(values, 1):
            rows.append([diameter, rh, j / 2, gf, sd])
    assert len(rows) == 294 and len({tuple(v[:3]) for v in rows}) == 294
    save(output / "table-s1.csv", "diameter_nm,rh_pct,time_h,gf,spread", rows)


def table_s2(supp, output):
    table = (
        subprocess.check_output(["pdftotext", "-layout", supp, "-"])
        .decode()
        .rsplit("Table S2.", 1)[1]
    )
    observations = []
    for line in table.splitlines():
        values = re.findall(r"(\d[.]\d+)\s*±\s*(\d[.]\d+)", line)
        if not values:
            continue
        assert len(values) == 3
        keys = line[: line.index(values[0][0])].split()
        if len(keys) == 2:
            ovf = int(keys[0])
        rh = int(keys[-1])
        observations.extend(
            [ovf, rh, hour, gf, spread]
            for hour, (gf, spread) in zip([0, 2, 3], values)
        )
    assert (
        len(observations)
        == len({tuple(obs[:3]) for obs in observations})
        == 108
    )
    save(
        output / "figure6-s2.csv",
        "ovf_pct,rh_pct,time_h,gf,spread",
        observations,
    )


def relative_changes(main, output):
    text = subprocess.check_output(["pdftotext", "-raw", main, "-"]).decode()
    paragraph = " ".join(text.split()).split(
        "Compared to fresh NaCl aerosols,", 1
    )[1]
    rows = []
    for humidity, direction, sign in [
        (60, "increased", 1),
        (80, "decreased", -1),
    ]:
        series = paragraph.split(f"{direction} by ", 1)[1].split(
            " after reaction", 1
        )[0]
        percentages = re.findall(r"(\d+(?:\.\d+)?)\s*%", series)
        if len(percentages) != 5:
            raise ValueError(
                f"Expected five percentage changes at {humidity}% RH"
            )
        for hour, value in enumerate(percentages, 1):
            rows.append([humidity, hour, format(sign * float(value), "g")])
    save(
        output / "text-100nm.csv",
        "rh_pct,time_h,reported_percent_change_from_fresh",
        rows,
    )
