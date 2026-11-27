"""Compose independently supervised water and organic transfer networks."""

import argparse
from pathlib import Path

import torch
from stage2_transfer import Transfer


class Specialists(Transfer):
    """Compose independently trained water and organic networks."""

    def __init__(self, spec, water, organic, order):
        """Bind the two specialists and their integration order."""
        super().__init__(spec)
        self.water = self.install(spec, water, True)
        self.organic = self.install(spec, organic, False)
        self.ordering = order
        self.register_buffer(
            "water_lanes", torch.as_tensor([k == "water" for k in spec.names])
        )

    @staticmethod
    def install(spec, cp, is_water):
        """Restore one specialist and verify ownership of its vapor lanes."""
        if cp.get("active") != ("water" if is_water else "organic"):
            raise ValueError("Checkpoint was fitted to a different process")
        if cp.get("radau"):
            from stage2_radau import Radau

            net = Radau(spec, *cp["radau"], cp["kind"], cp["width"])
        else:
            net = Transfer(spec, cp["kind"], cp["width"])
        net.head.load_state_dict(cp["head"])
        return net

    def start(self, number):
        """Allocate independent recurrent memories for both processes."""
        return torch.cat(
            (self.water.start(number), self.organic.start(number)), -1
        )

    def step(self, q, g, st, dt, h):
        """Apply both specialists to a shared particle state."""
        nets = (self.water, self.organic)
        self.water.active_lanes = self.water_lanes
        self.organic.active_lanes = ~self.water_lanes
        hs = list(torch.split(h, [net.memory_width for net in nets], dim=-1))
        schedules = {
            "water-first": ((0, 1.0), (1, 1.0)),
            "organic-first": ((1, 1.0), (0, 1.0)),
            "symmetric": ((0, 0.5), (1, 1.0), (0, 0.5)),
        }
        for i, f in schedules[self.ordering]:
            q, g, new = nets[i].step(q, g, st, dt * f, hs[i])
            hs[i] = torch.where(
                nets[i].active_lanes[None, None, :, None], new, hs[i]
            )
        return q, g, torch.cat(hs, -1)


def pack(water, organic, out, order):
    """Package the two fitted networks and the deployed integration order."""
    a, b = [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in (water, organic)
    ]
    if a["active"] != "water" or b["active"] != "organic":
        raise ValueError("Each specialist needs its own physical target")
    torch.save(dict(specialists=True, water=a, organic=b, order=order), out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("water", "organic", "out"):
        p.add_argument(k, type=Path)
    p.add_argument(
        "--order",
        choices=["symmetric", "water-first", "organic-first"],
        default="symmetric",
    )
    pack(**vars(p.parse_args()))
