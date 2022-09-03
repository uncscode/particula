""" partycula.party """

from collections import Counter


class PreGame:
    """ Attendance """
    def __init__(self, radius):
        self.radius = radius if isinstance(radius, list) else [radius]

    def __repr__(self):
        return (
            f"radius (m): {[f'{i:.2e}' for i in Counter(self.radius).keys()]}\n"
            f"counts (#): {[f'{i:.2e}' for i in Counter(self.radius).values()]}"
        )

    def __add__(self, other):  # self + other
        return PreGame(radius=self.radius + other.radius)

    def __radd__(self, other):  # other + self
        return PreGame(radius=other.radius + self.radius)

    def __sub__(self, other):  # self - other
        orig = self.radius.copy()
        for i in [j for j in other.radius if j in orig]:
            _ = orig.remove(i) if i in orig else orig
        return PreGame(radius=orig)

    def __rsub__(self, other):  # other - self
        orig = other.radius.copy()
        for i in [j for j in self.radius if j in orig]:
            _ = orig.remove(i) if i in orig else orig
        return PreGame(radius=orig)


class Party(PreGame):
    """ Party """
    pass
