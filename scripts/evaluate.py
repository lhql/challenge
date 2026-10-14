#!/usr/bin/env python3
"""Receding policy evaluation in rigid-body simulator, explicitly NOT PX4 flight."""

import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package))
from gap_nav.learning.networks import Actor
from gap_nav.data import render_observation
from gap_nav.dynamics import rollout, initial_memory, rotation, Physics
from gap_nav.trajectory import references
from gap_nav.costs import clearance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--cases", type=int, default=16)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--data", type=Path, default=package / "results/data/samples.pt")
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=True)
    cfg = Physics(**checkpoint.get("physics", {}))
    actor = Actor(checkpoint["sensor"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    data = torch.load(a.data, map_location=device, weights_only=True)
    valid = (~data["train"]).nonzero().flatten()
    ids = valid[:: max(1, len(valid) // a.cases)][: a.cases]
    x = data["x"][ids]
    goal = data["goal"][ids]
    boxes = data["boxes"][data["map_id"][ids]]
    initial = x.clone()
    act = initial_memory(x, cfg)
    command = act.clone()
    collided = torch.zeros(len(x), dtype=torch.bool, device=device)
    success = collided.clone()
    duration = torch.zeros(len(x), device=device)
    travel = duration.clone()
    min_distance = duration + 100
    tilt_max = duration.clone()
    rate_max = duration.clone()
    jerk_sum = duration.clone()
    last_acc = torch.zeros_like(x[:, :3])
    paths = [x.cpu()]
    latencies = []
    with torch.no_grad():
        for t in range(a.steps):
            active = ~(collided | success)
            if not bool(active.any()):
                break
            image = render_observation(
                x, boxes, checkpoint["sensor"], package.parents[1]
            )
            if device == "cuda":
                torch.cuda.synchronize()
            before = time.perf_counter()
            knots = actor(image, x, goal)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - before) * 1000)
            ref = references(x, knots)[:, :1]
            wind = torch.randn(len(x), 1, 3, device=device) * 0.06
            X, U, M = rollout(x, ref, wind, (act, command), cfg)
            next_x = X[:, 1]
            act = torch.where(active[:, None], M[:, 1, 0], act)
            command = torch.where(active[:, None], M[:, 1, 1], command)
            travel += torch.where(active, (next_x[:, :3] - x[:, :3]).norm(dim=-1), 0)
            duration += active.float() * 0.08
            acceleration = (next_x[:, 3:6] - x[:, 3:6]) / 0.08
            jerk_sum += torch.where(
                active, ((acceleration - last_acc) / 0.08).norm(dim=-1) * 0.08, 0
            )
            last_acc = acceleration
            x = torch.where(active[:, None], next_x, x)
            distance = clearance(x[:, None, :3], boxes).squeeze(-1) - cfg.radius
            min_distance = torch.minimum(min_distance, distance)
            R = rotation(x[:, 6:10])
            tilt_max = torch.maximum(tilt_max, torch.acos(R[:, 2, 2].clamp(-1, 1)))
            rate_max = torch.maximum(rate_max, x[:, 10:].norm(dim=-1))
            collided |= distance <= 0
            success |= (
                ((x[:, :3] - goal).norm(dim=-1) < 0.35)
                & (x[:, 3:6].norm(dim=-1) < 0.25)
                & ~collided
            )
            paths.append(x.cpu())
    report = dict(
        kind="offline rigid-body receding policy; NOT PX4 Gazebo",
        sensor=checkpoint["sensor"],
        cases=len(x),
        success_rate=float(success.float().mean()),
        collision_rate=float(collided.float().mean()),
        mean_speed=float((travel / duration.clamp_min(0.08)).mean()),
        mean_episode_s=float(duration.mean()),
        minimum_clearance_m=float(min_distance.min()),
        max_tilt_rad=float(tilt_max.max()),
        max_angular_rate_rad_s=float(rate_max.max()),
        mean_jerk=float((jerk_sum / duration.clamp_min(0.08)).mean()),
        batch_inference_ms=float(np.mean(latencies)),
        eval_wall_s=time.perf_counter() - start,
        ids=ids.cpu().tolist(),
        max_steps=a.steps,
        initial_distance=(initial[:, :3] - goal).norm(dim=-1).cpu().tolist(),
        final_distance=(x[:, :3] - goal).norm(dim=-1).cpu().tolist(),
        success=success.cpu().tolist(),
        collision=collided.cpu().tolist(),
    )
    a.checkpoint.with_suffix(".evaluation.json").write_text(
        json.dumps(report, indent=2)
    )
    torch.save(torch.stack(paths), a.checkpoint.with_suffix(".rollout.pt"))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
