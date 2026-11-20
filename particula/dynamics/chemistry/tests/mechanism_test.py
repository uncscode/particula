"""Native chemistry interchange and binding, independent of integration."""

import numpy as np
import pytest
from particula.dynamics.chemistry import ReactionMechanism
from particula.gas.gas_data import GasData


def conversion():
    """Construct a bimolecular conversion."""
    return {
        "version": 1,
        "basis": "mol/m^3-air",
        "species": {"gas:A": {}, "gas:B": {}, "gas:C": {}},
        "reactions": [
            {
                "id": "combine",
                "reactants": {"gas:A": 1, "gas:B": 1},
                "products": {"gas:C": 1},
                "rate": {"type": "constant", "k": 2, "units": "m^3/mol/s"},
            }
        ],
    }


def gas():
    """Build a permuted layout with unequal masses."""
    return GasData(
        ["C", "A", "precursor", "inert"],
        np.array([0.03, 0.01, 0.02, 1]),
        np.zeros((2, 4)),
        np.zeros(4, dtype=bool),
    )


@pytest.mark.parametrize("extension", ["json", "yaml", "yml"])
def test_interchange(tmp_path, extension):
    """Round-trip without retaining mutable caller inputs."""
    record = conversion()
    mech = ReactionMechanism.from_dict(record)
    path = tmp_path / f"network.{extension}"
    mech.save(path)
    assert ReactionMechanism.load(path) == mech
    record["reactions"].clear()
    assert len(mech.to_dict()["reactions"]) == 1


@pytest.mark.parametrize(
    ("ext", "data"),
    [
        ("json", '{"version":1,"version":2}'),
        ("yaml", "version: 1\nversion: 2\n"),
    ],
)
def test_duplicate_fields(tmp_path, ext, data):
    """Reject duplicate fields during parsing."""
    p = tmp_path / f"bad.{ext}"
    p.write_text(data)
    with pytest.raises(ValueError, match="duplicate key"):
        ReactionMechanism.load(p)


def test_binding_and_jacobian_at_zero():
    """Resolve renamed species and check zero-state derivatives."""
    mech = ReactionMechanism(conversion())
    g = gas()
    bound = mech.bind(g, {"gas:B": "precursor"})
    assert bound.indices == (1, 2, 0)
    f = bound.kinetics(298)
    np.testing.assert_array_equal(
        f.rate(0, np.array([2.0, 3.0, 0.0])), [-12, -12, 12]
    )
    np.testing.assert_array_equal(
        f.jacobian(0, np.array([0.0, 3.0, 0.0])),
        [[-6, 0, 0], [-6, 0, 0], [6, 0, 0]],
    )
    g.name[0], g.name[1] = g.name[1], g.name[0]
    with pytest.raises(ValueError, match="rebind"):
        bound.validate(g)


def test_binding_rejections():
    """Reject incompatible names, masses, or aliases."""
    mech = ReactionMechanism(conversion())
    with pytest.raises(ValueError, match="unmapped"):
        mech.bind(gas())
    with pytest.raises(ValueError, match="same gas column"):
        mech.bind(gas(), {"gas:B": "A"})
    g = gas()
    g.molar_mass[2] += 0.01
    with pytest.raises(ValueError, match="molar masses"):
        mech.bind(g, {"gas:B": "precursor"})


def test_invalid_chemistry():
    """Reject rate unit and elemental balance errors."""
    record = conversion()
    record["reactions"][0]["rate"]["units"] = "1/s"
    with pytest.raises(ValueError, match="combine.*units"):
        ReactionMechanism(record)
    record = conversion()
    for meta in record["species"].values():
        meta["elements"] = {"H": 1}
    with pytest.raises(ValueError, match="elemental"):
        ReactionMechanism(record).bind(gas(), {"gas:B": "precursor"})
