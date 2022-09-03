""" partycula.party """

from collections import Counter


class Party:
    """ a party of partycles """
    def __init__(self, radius):
        self.radius = radius if isinstance(radius, list) else [radius]

    def __repr__(self):
        return (
            f"radius: {list(Counter(self.radius).keys())}\n"
            f"counts: {list(Counter(self.radius).values())}"
        )

    def __add__(self, other):  # self + other
        return Party(radius=self.radius + other.radius)

    def __radd__(self, other):  # other + self
        return Party(radius=other.radius + self.radius)

    def __sub__(self, other):  # self - other
        orig = self.radius.copy()
        for i in [j for j in other.radius if j in orig]:
            _ = orig.remove(i) if i in orig else orig
        return Party(radius=orig)

    def __rsub__(self, other):  # other - self
        orig = other.radius.copy()
        for i in [j for j in self.radius if j in orig]:
            _ = orig.remove(i) if i in orig else orig
        return Party(radius=orig)
