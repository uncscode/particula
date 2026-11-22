"""Build Manchester reaction proposals using SiMBA balance rules."""

from dataclasses import asdict
from itertools import combinations_with_replacement

from baseline_replay import BANK, CANDIDATES, TRIALS, Channel
from particula.dynamics.chemistry.discovery import balanced_steps


def balanced_accretion():
    """Enumerate balanced pairs preserving the three precursor labels."""
    # G, V and P preserve one parent equivalent; D preserves the incoming pair.
    # The gas route includes prompt condensation of its accretion product.
    # Products with no parent-specific storage are excluded from this generator.
    labels = ["A", "C", "I"]
    pairs = {
        (0, 0): (11, "AA"),
        (0, 1): (12, "AC"),
        (1, 1): (14, "CC"),
        (2, 2): (15, "II"),
    }
    for phase, sources in (("gas", range(6)), ("particle", range(6, 9))):
        sources = list(sources)
        composition = [
            tuple(int(j == s % 3) for j in range(3)) for s in sources
        ]
        for (a, b), (dest, label) in pairs.items():
            basis = tuple(int(j == a) + int(j == b) for j in range(3))
            steps = set(
                balanced_steps(
                    [phase] * (len(sources) + 1), composition + [basis]
                )
            )
            for i, j in combinations_with_replacement(range(len(sources)), 2):
                row = tuple(
                    -int(k == i) - int(k == j) for k in range(len(sources))
                ) + (1,)
                if row not in steps:
                    continue
                left, right = sources[i], sources[j]

                def name(s):
                    return ("G", "V", "P")[s // 3] + "_" + labels[s % 3]

                yield Channel(
                    f"accrete_{left}_{right}",
                    f"{name(left)} + {name(right)} → D_{label}",
                    f"{phase} accretion of parent-tagged oxygenates",
                    left,
                    right,
                    dest,
                )


def bank_and_motifs():
    """Extend the archived reaction bank and keep unique proposal paths."""
    bank = list(BANK)
    motifs = [tuple(t["ids"]) for t in TRIALS]

    def identity(r):
        sources = (r.a,) if r.b < 0 else tuple(sorted((r.a, r.b)))
        return sources, r.dest, r.split, r.area_scaled

    known = {identity(r): i for i, r in enumerate(bank)}

    def add(r):
        idx = known.get(identity(r))
        if idx is None:
            idx = len(bank)
            bank.append(r)
            known[identity(r)] = idx
        motifs.append((idx,))
        return idx

    for r in CANDIDATES:
        if r.key not in {channel.key for channel in bank}:
            add(Channel(**asdict(r)))
    for reaction in balanced_accretion():
        add(reaction)
    # Reversible release and recapture of a labelled family's secondary vapor.
    for j, label in enumerate(("AA", "AC", "AU", "CC", "II")):
        release_and_return = []
        for src, dst, eq, area in [
            (16 + j, 26 + j, "W → U", False),
            (21 + j, 26 + j, "L → U", False),
            (26 + j, 21 + j, "U → L", True),
        ]:
            channel_id = add(
                Channel(
                    f"{label}_{src}_{dst}",
                    f"{label}: {eq}",
                    "Secondary vapor release or area-dependent return",
                    src,
                    -1,
                    dst,
                    area,
                )
            )
            if src == 16 + j or area:
                release_and_return.append(channel_id)
        motifs.append(tuple(release_and_return))
    return bank, list(dict.fromkeys(motifs))
