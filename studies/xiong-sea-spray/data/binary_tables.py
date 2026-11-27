"""Recover bulk water activities and H-TDMA measurements from publisher workbooks."""

import csv
import io
import math
from xml.etree import ElementTree
from zipfile import ZipFile

SOS_KG_PER_MOL = 0.23227
WATER_KG_PER_MOL = 0.01801528
XML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def first_sheet(data):
    """Each pinned workbook stores the source data in sheet1.xml."""
    with ZipFile(io.BytesIO(data)) as book:
        strings = [
            "".join(node.itertext())
            for node in ElementTree.fromstring(
                book.read("xl/sharedStrings.xml")
            )
        ]
        root = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
        cells = {}
        for cell in root.iter(XML + "c"):
            val = cell.findtext(XML + "v")
            if val is not None:
                cells[cell.get("r")] = (
                    strings[int(val)] if cell.get("t") == "s" else float(val)
                )
        return cells


def mean_sd(value):
    return tuple(float(part.strip()) for part in value.split("±"))


def shahabadi(cells):
    if cells["A20"] != "SOS" or cells["D1"] != "Water activity":
        raise ValueError("Unexpected Shahabadi Fig. 2 sheet or SOS block")
    for row in (22, 21, 20, 19):
        c, rho, aw = (cells[f"{col}{row}"] for col in "BCD")
        m = c / (rho - SOS_KG_PER_MOL * c)
        water_ratio = 1 / (m * WATER_KG_PER_MOL)
        yield dict(
            study="Shahabadi2025",
            source_row=row,
            T_K=295,
            c_SOS_mol_L=c,
            rho_kg_L=rho,
            m_SOS_mol_kg_water=m,
            w_SOS=SOS_KG_PER_MOL * c / rho,
            mol_H2O_per_mol_SOS=water_ratio,
            mol_H2O_per_mol_SOS_sd="",
            a_w=aw,
            a_w_meter_accuracy=0.003,
            phi_apparent_nu2=-math.log(aw) * water_ratio / 2,
        )


def peng_water(cells):
    if cells["B12"] != "sodium octyl sulfate" or cells["C14"] != "WSR":
        raise ValueError("Unexpected Peng Table 1 chemical or WSR column")
    ratio, sd = mean_sd(cells["C23"])
    aw = cells["A23"] / 100
    water_kg = ratio * WATER_KG_PER_MOL
    return dict(
        study="Peng2021",
        source_row=23,
        T_K=298.15,
        c_SOS_mol_L="",
        rho_kg_L="",
        m_SOS_mol_kg_water=1 / water_kg,
        w_SOS=SOS_KG_PER_MOL / (SOS_KG_PER_MOL + water_kg),
        mol_H2O_per_mol_SOS=ratio,
        mol_H2O_per_mol_SOS_sd=sd,
        a_w=aw,
        a_w_meter_accuracy="",
        phi_apparent_nu2=-math.log(aw) * ratio / 2,
    )


def peng_aerosol(cells):
    if cells["D1"] != "Sodium octyl" or cells["D2"] != "sulfate":
        raise ValueError("Unexpected Peng Table 2 chemical")
    for row in range(3, 15):
        gf, sd = mean_sd(cells[f"D{row}"])
        yield dict(
            source_row=row,
            RH_percent=cells[f"A{row}"],
            dry_mobility_nm=100,
            T_K=297.15,
            GF=gf,
            GF_sd=sd,
        )


def write_table(path, rows):
    precision = {
        "m_SOS_mol_kg_water": 7,
        "w_SOS": 7,
        "mol_H2O_per_mol_SOS": 6,
        "phi_apparent_nu2": 6,
        "a_w": 6,
        "rho_kg_L": 6,
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            for col, digits in precision.items():
                if col in row and row[col] != "":
                    row[col] = round(row[col], digits)
            if "RH_percent" in row:
                row["RH_percent"] = int(row["RH_percent"])
            writer.writerow(row)


def binary_tables(cache, output):
    with ZipFile(cache / "shahabadi-2025-supplement.zip") as archive:
        shah = first_sheet(archive.read("Surface-tension-and-RI.xlsx"))
    water = list(shahabadi(shah))
    water.append(
        peng_water(first_sheet((cache / "peng-2021-t01.xlsx").read_bytes()))
    )
    aerosols = list(
        peng_aerosol(first_sheet((cache / "peng-2021-t02.xlsx").read_bytes()))
    )
    write_table(output / "binary-water.csv", water)
    write_table(output / "binary-aerosol-check.csv", aerosols)
