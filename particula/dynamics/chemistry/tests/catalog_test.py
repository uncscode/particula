"""Source parsing, immutable selections and native atmospheric rate oracles."""

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from math import exp

import pytest
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.catalog import ChemistryCatalog, sources
from particula.dynamics.chemistry.catalog.cli import main
from particula.dynamics.chemistry.catalog.cmc import read_cmc
from particula.dynamics.chemistry.catalog.readers import ro2_members

MOZART = "Reactions\n[t] NO + O3 -> NO2 + O2 ; 3e-12,-1500\nEnd Reactions\n"
DECK = """Description | molecular units
Num Reactions | 1
Reaction Records
Rxn Typ Order
1 3 2 2.070D-12 1400 0.00 300
"""


@pytest.mark.parametrize(
    ("source", "source_text", "a", "e"),
    [
        ("waccm", MOZART, 3e-12, 1500),
        ("e3sm", "Fixed\nO2\nEnd Fixed\n" + MOZART, 3e-12, 1500),
        (
            "wrf-chem",
            "#EQUATIONS\n{T} NO + O3 = NO2{ + O2} : ARR2(3e-12_dp,1500._dp,TEMP);",
            3e-12,
            1500,
        ),
        (
            "cmaq",
            "REACTIONS[CM] =\n<R> NO + O3 = NO2 # 3e-12 @ 1500.;\nEND MECH",
            3e-12,
            1500,
        ),
        (
            "mcm",
            "#EQUATIONS\n<7> NO + O3 = NO2 : 1.4e-12*EXP(-1310./TEMP);",
            1.4e-12,
            1310,
        ),
        ("camx", DECK, 2.07e-12, 1400),
        ("pmcamx", DECK.replace("2.070D-12 1400", "3D-12 1500"), 3e-12, 1500),
    ],
)
def test_gas_dialects(tmp_path, source, source_text, a, e):
    """Evaluate seven source representations through the same native rates."""
    file = tmp_path / "source"
    file.write_text(source_text)
    listing = tmp_path / "mech"
    listing.write_text("1 O3 NO = 1.0 NO2\n")
    companions = [listing] if source in {"camx", "pmcamx"} else []
    c = ChemistryCatalog(tmp_path / "db")
    snap = c.import_file(
        source, file, release="v1", mechanism="example", companions=companions
    )
    (row,) = c.search(species="NO", role="reactant")
    assert row["id"].startswith(snap + "/")
    output = tmp_path / "native.json"
    if source != "waccm":
        with pytest.raises(ValueError, match="close-oxygen"):
            c.export([row["id"]], output)
    lock = tmp_path / "lock.json"
    c.export([row["id"]], lock, target="lock", close_oxygen=True)
    ChemistryCatalog(tmp_path / "empty").export([], output, lock_path=lock)
    record = ReactionMechanism.load(output).to_dict()["reactions"][0]
    assert record["products"] == {"gas:NO2": 1, "gas:O2": 1}
    rate = record["rate"]
    for temperature in (250.0, 298.0, 320.0):
        k = rate["A"] * exp(
            -rate["Ea_J_per_mol"] / (8.31446261815324 * temperature)
        )
        assert k == pytest.approx(
            a * 6.02214076e17 * exp(-e / temperature), rel=1e-8
        )


def test_snapshots_and_rollback(tmp_path):
    """Cold and simultaneous imports converge and corrections retain old rates."""
    c = ChemistryCatalog(tmp_path / "db")
    file = tmp_path / "input"
    file.write_text(MOZART)

    def ingest():
        return c.import_file("waccm", file, release="1", mechanism="model")

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.map(lambda _: ingest(), range(2))
    assert a == b
    original = c.show(a + "/t")
    file.write_text("\n" + MOZART)
    assert c.show(ingest() + "/t")["revision"] == original["revision"]
    file.write_text(MOZART.replace("3e-12", "4e-12"))
    updated = c.show(ingest() + "/t")
    assert updated["revision"] != original["revision"]
    assert c.show(a + "/t")["kinetics"][0] == 3e-12
    with pytest.raises(ValueError, match="overlapping"):
        c.export([original["id"], updated["id"]], tmp_path / "test.json")
    file.write_text(MOZART + MOZART)
    with pytest.raises(ValueError, match="duplicate"):
        ingest()
    assert len(c.search()) == 3
    assert c.source_bytes(hashlib.sha256(file.read_bytes()).hexdigest()) is None


def test_unsupported_and_corrupt_lock_preserve_output(tmp_path):
    """Photolysis forcing and altered locks fail before replacing a study."""
    c = ChemistryCatalog(tmp_path / "db")
    file = tmp_path / "photo"
    file.write_text("Photolysis\n[j] NO2 + hv -> NO + O\nEnd Photolysis")
    s = c.import_file("waccm", file, release="1", mechanism="model")
    out = tmp_path / "study.json"
    out.write_text("previous")
    with pytest.raises(ValueError, match="photolysis"):
        c.export([s], out)
    lock = tmp_path / "lock"
    c.export([s], lock, target="lock")
    lock.write_text(
        lock.read_text().replace(
            '"close_oxygen": false', '"close_oxygen": true'
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        c.export([], out, lock_path=lock)
    assert out.read_text() == "previous"


def test_fetch_and_cli(tmp_path, monkeypatch, capsys):
    """Rebuild a verified recipe, then query without network or source files."""
    monkeypatch.setenv("PARTICULA_CHEM_CATALOG", str(tmp_path / "db"))
    raw = MOZART.encode()
    recipe = {
        "source": "waccm",
        "release": "r",
        "mechanism": "trial",
        "aerosol": None,
        "files": [
            dict(
                name="mech.in",
                url="https://example.org/mech",
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        ],
    }
    monkeypatch.setattr(sources, "recipes", lambda: {"waccm": recipe})
    monkeypatch.setattr(
        sources, "urlopen", lambda *a, **kw: io.BytesIO(raw + b"!")
    )
    with pytest.raises(ValueError, match="checksum"):
        ChemistryCatalog().fetch("waccm")
    monkeypatch.setattr(sources, "urlopen", lambda *a, **kw: io.BytesIO(raw))
    main(["chem", "fetch", "waccm", "--json"])
    snapshot = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(
        sources, "urlopen", lambda *a, **kw: pytest.fail("offline")
    )
    assert ChemistryCatalog().fetch("waccm") == snapshot["id"]
    main(["chem", "search", "--species", "NO2", "--role", "product", "--json"])
    (row,) = json.loads(capsys.readouterr().out)
    main(["chem", "show", row["id"], "--json"])
    assert json.loads(capsys.readouterr().out)["implementation"].endswith(
        "ReactionMechanism.bind"
    )
    main(["chem", "export", row["id"], "--output", str(tmp_path / "gas.json")])
    assert (tmp_path / "gas.json").exists()


def test_cmc_and_ro2():
    """Check number alignment, stoichiometry, kinetic orders and radical pool."""
    sx = "      subroutine lsrxn1()\n      r(3)=dbrk(3)*y(lO3)*y(lNO)"
    rate = "      subroutine lsrate1()\n      Loss(lNO)= + r(3)\n      Loss(lO3)= + r(3)\n      Gain(lNO2)= + r(3)"
    assert read_cmc([sx, rate])["3"] == (
        {"NO": 1, "O3": 1},
        {"NO2": 1},
        {"O3": 1, "NO": 1},
    )
    with pytest.raises(ValueError, match="matching"):
        read_cmc([sx, rate.replace("lsrate1", "lsrate7")])
    assert ro2_members(
        "other=C(ind_Z)\n RO2 = C(ind_A) + &\n C(ind_B)\n#EQUATIONS"
    ) == ["A", "B"]


def test_mcm_renumbering(tmp_path):
    """Match chemistry across York subset exports with different row labels."""
    c = ChemistryCatalog(tmp_path / "db")
    file = tmp_path / "subset"

    def row(label):
        file.write_text(
            f"#EQUATIONS\n<{label}> NO + O3 = NO2 : 1.4e-12*EXP(-1310./TEMP);"
        )
        s = c.import_file("mcm", file, release="3.3.1", mechanism="subset")
        return c.show(s + "/" + label)

    first, last = row("1"), row("900")
    assert first["id"] != last["id"]
    assert first["revision"] == last["revision"]


def test_soap_yields_and_aging(tmp_path):
    """Preserve both NOx endpoints and the alternative biomass-bin coefficient."""
    template = """      subroutine soachem()
      data rkpar / 8.1d-13,0.0 / ! rxn 1: PAR + OH
      data lnoxval / 0.0,0.0046,0.1,0.0 / ! rxn 1
      data hnoxval / 0.0,0.0023,0.,0.2 / ! rxn 1
      data idCG / 2,3,4,5 / ! rxn 1
      dcCG(19) = 0
c-----Calculate and Apply Gain and Loss
      do iorg = 19,25
        cCG(iorg) = cCG(iorg) - dcCG(iorg) + (FACTOR) * dcCG(iorg+1)
      enddo
"""
    c = ChemistryCatalog(tmp_path / "db")
    file = tmp_path / "soachem.f"
    for factor in (1.075, 1.0):
        file.write_text(template.replace("FACTOR", str(factor)))
        snap = c.import_file("pmcamx", file, release="test", mechanism="SOAP")
        precursor = c.show(snap + "/SOAP1")
        assert precursor["metadata"]["low_nox_yields"][1] == 0.0046
        assert precursor["metadata"]["high_nox_yields"][1] == 0.0023
        assert (
            c.search(snapshot=snap, species="CAS3", role="product")[0]["id"]
            == precursor["id"]
        )
        assert precursor["metadata"]["yield_bins"] == [
            "CAS2",
            "CAS3",
            "CAS4",
            "CAS5",
        ]
        assert precursor["equation"] == "PAR + OH -> CAS3, CAS4, CAS5"
        assert c.search(snapshot=snap, species="CAS2", role="product") == []
        for species in ("CAS4", "CAS5"):
            assert (
                c.search(snapshot=snap, species=species, role="product")[0][
                    "id"
                ]
                == precursor["id"]
            )
        aging = c.show(snap + "/aging-20-19")
        assert aging["products"] == {"CG19": factor}
        assert aging["metadata"]["host_rate"] == "rkpoa"
        with pytest.raises(ValueError, match="prescribed OH"):
            c.export([aging["id"]], tmp_path / "aging.json")


def test_new_reader_keeps_previous_snapshot(tmp_path, monkeypatch):
    """Search uses the latest reader for identical inputs and can replay old IDs."""
    from particula.dynamics.chemistry.catalog import store

    catalog = ChemistryCatalog(tmp_path / "db")
    file = tmp_path / "case"
    file.write_text(MOZART)
    first = catalog.import_file("waccm", file, release="test", mechanism="box")
    monkeypatch.setattr(store, "IMPORT_VERSION", store.IMPORT_VERSION + 1)
    second = catalog.import_file("waccm", file, release="test", mechanism="box")
    assert first != second
    assert [row["id"] for row in catalog.search()] == [second + "/t"]
    assert catalog.search(snapshot=first)[0]["id"] == first + "/t"


@pytest.mark.parametrize(
    ("source", "tag"), [("mcm", "<R>"), ("wrf-chem", "{R}")]
)
@pytest.mark.parametrize("location", ["primary", "companion"])
def test_kpp_fixed_reservoir(tmp_path, source, tag, location):
    """A fixed OH product precludes closed gas export in both KPP dialects."""
    fixed = "  #DEFFIX\nH2O = IGNORE; OH = H + O; {NO = ignored comment}\n"
    file = tmp_path / "case.eqn"
    equation = f"#EQUATIONS\n{tag} NO + HO2 = NO2 + OH : 1e-12;\n"
    file.write_text(equation + (fixed if location == "primary" else ""))
    companion = tmp_path / "case.spc"
    companion.write_text(fixed if location == "companion" else "")
    catalog = ChemistryCatalog(tmp_path / "db")
    snap = catalog.import_file(
        source, file, release="1", mechanism="box", companions=[companion]
    )
    fixed_species = catalog.show(snap + "/R")["metadata"]["fixed_species"]
    assert "OH" in fixed_species and "NO" not in fixed_species
    with pytest.raises(ValueError, match="prescribed reservoir"):
        catalog.export([snap], tmp_path / "gas.json")


@pytest.mark.parametrize(
    ("source", "tag"), [("mcm", "<R>"), ("wrf-chem", "{R}")]
)
def test_kpp_commented_equations(tmp_path, source, tag):
    """Keep actual labels, source lines and implicit oxygen through comments."""
    file = tmp_path / "case.eqn"
    file.write_text(
        "{ preamble\n#EQUATIONS\nignored; }\n#EQUATIONS\n"
        "{ mechanism note;\n#SETFIX OH; }\n"
        f"{tag} NO + O3 = NO2 {{ + O2}} : 1e-12; {{ trailing note; }}\n"
    )
    catalog = ChemistryCatalog(tmp_path / "db")
    snap = catalog.import_file(source, file, release="1", mechanism="box")
    row = catalog.show(snap + "/R")
    assert row["line"] == 7
    assert row["equation"] == "NO + O3 = NO2 { + O2}"
    assert "OH" not in row["metadata"]["fixed_species"]
    with pytest.raises(ValueError, match="closed-box O2"):
        catalog.export([snap], tmp_path / "gas.json")
    catalog.export([snap], tmp_path / "gas.json", close_oxygen=True)


@pytest.mark.parametrize(
    ("source", "tag"), [("mcm", "<R>"), ("wrf-chem", "{R}")]
)
@pytest.mark.parametrize("override_file", ["primary", "companion"])
def test_kpp_reclassifies_reservoir(tmp_path, source, tag, override_file):
    """The last explicit species setting controls closed-box export."""
    catalog = ChemistryCatalog(tmp_path / "db")
    file, companion = tmp_path / "case.eqn", tmp_path / "case.spc"
    equation = f"#EQUATIONS\n{tag} NO + HO2 = NO2 + OH : 1e-12;\n"
    declarations = "#DEFVAR\nOH = H + O;\n"
    for variable in (False, True):
        overrides = "#SETFIX OH;\n" + ("#SETVAR\n OH;\n" if variable else "")
        primary = overrides if override_file == "primary" else declarations
        companion.write_text(
            declarations if override_file == "primary" else overrides
        )
        file.write_text(equation + primary)
        snap = catalog.import_file(
            source, file, release="1", mechanism="box", companions=[companion]
        )
        assert (
            "OH" in catalog.show(snap + "/R")["metadata"]["fixed_species"]
        ) != variable
        if variable:
            catalog.export([snap], tmp_path / "gas.json")
        else:
            with pytest.raises(ValueError, match="prescribed reservoir"):
                catalog.export([snap], tmp_path / "gas.json")


def test_mcm_inactive_element_declarations(tmp_path):
    """Inactive atom declarations cannot make an unbalanced reaction exportable."""
    file = tmp_path / "case.eqn"
    file.write_text(
        "#DEFVAR\nA = C + H;\nB = C;\n{ old composition\nA = C;\n}\n"
        "#EQUATIONS\n<R> A = B : 2.0;\n"
    )
    catalog = ChemistryCatalog(tmp_path / "db")
    snap = catalog.import_file("mcm", file, release="1", mechanism="box")
    assert catalog.show(snap + "/R")["metadata"]["elements"]["A"] == {
        "C": 1,
        "H": 1,
    }
    with pytest.raises(ValueError, match="external material budget"):
        catalog.export([snap], tmp_path / "gas.json")


def test_mcm_repeated_atoms(tmp_path):
    """Keep all hydrogens when testing a local MCM equation's atom balance."""
    file = tmp_path / "source.eqn"
    file.write_text(
        "#DEFVAR\nA = C + H + H;\nB = C + H;\n#EQUATIONS\n<1> A = B : 2.0;\n"
    )
    c = ChemistryCatalog(tmp_path / "db")
    snap = c.import_file("mcm", file, release="1", mechanism="box")
    assert c.show(snap + "/1")["metadata"]["elements"]["A"] == {"C": 1, "H": 2}
    with pytest.raises(ValueError, match="external material budget"):
        c.export([snap], tmp_path / "gas.json")


def test_mcm_pool_revisions(tmp_path, monkeypatch):
    """Pool membership affects actual RO2 dependencies, including definitions."""
    from particula.dynamics.chemistry.catalog import readers

    monkeypatch.setattr(
        readers,
        "_mcm_rates",
        lambda _: {"KRO2NO": "2.7e-12*EXP(360./TEMP)", "KPOOL": "1.2e-12*RO2"},
    )
    c = ChemistryCatalog(tmp_path / "db")
    file = tmp_path / "subset.eqn"

    def revisions(member):
        file.write_text(
            f"RO2 = C(ind_{member})\n#EQUATIONS\n"
            "<1> NO + HO2 = NO2 + OH : KRO2NO;\n"
            "<2> A = B : 1.2e-12*RO2;\n"
            "<3> A = B : KPOOL;\n"
        )
        snap = c.import_file("mcm", file, release="1", mechanism="subset")
        return [row["revision"] for row in c.search(snapshot=snap)]

    named, direct, indirect = revisions("A")
    after = revisions("B")
    assert named == after[0]
    assert direct != after[1]
    assert indirect != after[2]


@pytest.mark.parametrize("source", ["camx", "pmcamx"])
def test_unverified_deck_units(tmp_path, source):
    """Keep numeric source parameters with unresolved units in show and lock."""
    c = ChemistryCatalog(tmp_path / "db")
    file = tmp_path / "rates"
    file.write_text(DECK.replace("molecular units", "custom case"))
    listing = tmp_path / "mech"
    listing.write_text("1 NO HO2 = 1 NO2 1 OH\n")
    snap = c.import_file(
        source, file, release="1", mechanism="example", companions=[listing]
    )
    shown = c.show(snap + "/1")
    lock = tmp_path / "study.lock"
    c.export([snap], lock, target="lock")
    for record in (shown, json.loads(lock.read_text())["reactions"][0]):
        rate = record["rate_model"]
        assert rate["A"] == 2.07e-12
        assert rate["concentration_unit"] is None
        assert rate["time_unit"] is None
    with pytest.raises(ValueError, match="units require verification"):
        c.export([snap], tmp_path / "gas.json")


@pytest.mark.parametrize(
    ("description", "concentration", "time"),
    [
        ("CB05: ppm units", "ppm", "min"),
        ("not molecular units", None, None),
        ("ppm units; molecular units", None, None),
    ],
)
def test_camx_local_unit_evidence(tmp_path, description, concentration, time):
    """Report a declared mixing-ratio basis and withhold ambiguous units."""
    file = tmp_path / "rates"
    file.write_text(DECK.replace("molecular units", description))
    listing = tmp_path / "mech"
    listing.write_text("1 NO HO2 = 1 NO2 1 OH\n")
    c = ChemistryCatalog(tmp_path / "db")
    snap = c.import_file(
        "camx", file, release="local", mechanism="gas", companions=[listing]
    )
    record = c.selection([snap])["reactions"][0]
    assert record["rate_model"]["concentration_unit"] == concentration
    assert record["rate_model"]["time_unit"] == time
    assert record["metadata"]["source_rate_units"]["evidence"]
    with pytest.raises(
        ValueError, match="ppm/min rate conversion|units require verification"
    ):
        c.export([snap], tmp_path / "gas.json")


@pytest.mark.parametrize(
    ("primary", "companion", "issue"),
    [
        ("#SETVAR ALL_SPEC;", "", "unsupported KPP #SETVAR selector"),
        ("#SETVAR OH;", "#SETFIX OH;", "ambiguous KPP reclassification"),
    ],
)
def test_kpp_unknown_classification_order(tmp_path, primary, companion, issue):
    """Unsupported selectors and conflicting file settings fail explicitly."""
    file, species = tmp_path / "case.eqn", tmp_path / "case.spc"
    file.write_text("#EQUATIONS\n<R> NO + HO2 = NO2 + OH : 1e-12;\n" + primary)
    species.write_text(companion)
    catalog = ChemistryCatalog(tmp_path / "db")
    with pytest.raises(ValueError, match=issue):
        catalog.import_file(
            "mcm", file, release="1", mechanism="box", companions=[species]
        )
    assert catalog.search() == []
