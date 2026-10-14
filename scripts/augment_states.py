#!/usr/bin/env python3
"""Cache multiple dynamics/goals per existing rendered view without image copies."""

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package))
from gap_nav.dynamics import rotation
from gap_nav.costs import clearance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--states", type=int, default=8192)
    p.add_argument("--vertical-goal-std", type=float, default=0.15)
    p.add_argument("--reverse-goal-fraction", type=float, default=0.0)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(73)
    start = time.perf_counter()
    d = torch.load(a.input, weights_only=True)
    if a.states < len(d["x"]):
        raise ValueError("Requested states below source view count")
    ids = torch.arange(a.states) % len(d["x"])
    x = d["x"][ids].clone()
    map_id = d["map_id"][ids]
    boxes = d["boxes"][map_id]
    R = rotation(x[:, 6:10])
    n = len(ids)
    velocity = torch.randn(n, 3) * torch.tensor([0.8, 0.25, 0.12])
    velocity[:, 0] = (velocity[:, 0] + 0.6).clamp(0, 2.0)
    velocity[torch.rand(n) < 0.45] = 0
    x[:, 3:6] = (R @ velocity[:, :, None]).squeeze(-1)
    x[:, 10:] = torch.randn(n, 3) * 0.08
    relative = torch.zeros(n, 3)
    goal = x[:, :3].clone()
    pending = torch.ones(n, dtype=torch.bool)
    for attempt in range(200):
        j = pending.nonzero().flatten()
        if not len(j):
            break
        body = torch.stack(
            (
                torch.rand(len(j)) * 5 + 0.05,
                torch.randn(len(j)) * 0.8,
                torch.randn(len(j)) * a.vertical_goal_std,
            ),
            -1,
        )
        body[torch.rand(len(j)) < a.reverse_goal_fraction, 0] *= -1
        close = torch.rand(len(j)) < 0.3
        body[close] *= 0.2
        proposed = x[j, :3] + (R[j] @ body[:, :, None]).squeeze(-1)
        valid = (
            (clearance(proposed[:, None], boxes[j]).squeeze(-1) > 0.6)
            & (proposed[:, 2] > 0.65)
            & (proposed[:, 2] < 3.1)
        )
        goal[j[valid]] = proposed[valid]
        pending[j[valid]] = False
    if pending.any():
        raise RuntimeError("Could not sample free goal positions")
    d.update(x=x, goal=goal, image_id=ids, map_id=map_id, train=d["train"][ids])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, a.output)
    report = dict(
        source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        states=n,
        vertical_goal_std=a.vertical_goal_std,
        reverse_goal_fraction=a.reverse_goal_fraction,
        unique_views=len(d["depth"]),
        generation_s=time.perf_counter() - start,
        seed=73,
        note="Cached new dynamics and collision-free goals; perception/environment views reused exactly",
    )
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
