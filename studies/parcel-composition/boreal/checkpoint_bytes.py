"""Attribute actual restart array bytes to dry decades of the example
checkpoint.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def inventory(path):
    """Count the record, slot-index and integer-lineage buffers separately."""
    rows = []
    with np.load(path) as data:
        for i in range(2):
            prefix = f"p{i}_"
            centers = data[prefix + "centers"]
            info = data[prefix + "bin_count"]
            slots = data[prefix + "slots"]
            pending = data[prefix + "pending"]
            bits = data[prefix + "old_bits"]
            decade = (info[:, 0] - 1) // 10
            byte_owner = decade[slots[::8]]
            allocation = 0
            for j in range(-1, 4):
                records = decade == j
                members = records[slots]
                record_bytes = centers[records].nbytes + info[records].nbytes
                member_bytes = (
                    slots[members].nbytes
                    + pending[members].nbytes
                    + bits[byte_owner == j].nbytes
                )
                rows.append(
                    [
                        i,
                        j,
                        records.sum(),
                        members.sum(),
                        record_bytes,
                        member_bytes,
                        record_bytes + member_bytes,
                    ]
                )
                allocation += record_bytes + member_bytes
            np.testing.assert_equal(
                allocation,
                sum(
                    data[prefix + k].nbytes
                    for k in [
                        "centers",
                        "bin_count",
                        "slots",
                        "pending",
                        "old_bits",
                    ]
                ),
            )
    return pd.DataFrame(
        rows,
        columns=(
            "parcel decade rows members representative_bytes "
            "lineage_and_slot_bytes total_array_bytes"
        ).split(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    inventory(a.checkpoint).to_csv(a.out, index=False)
