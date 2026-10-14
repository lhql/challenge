"""Cached GPU training. No teacher, shadows, renderer or Gazebo inside updates."""

import argparse, hashlib, json, time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from .networks import Actor, Critic
from ..trajectory import trajectory
from ..costs import navigation_cost, clearance
from ..dynamics import Physics


def train(
    data_path,
    out,
    sensor="depth",
    epochs=30,
    batch_size=16,
    shadows=None,
    initial=None,
    max_speed=6.0,
    max_accel=4.0,
):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Physics(max_speed=max_speed, max_accel=max_accel)
    total_start = time.perf_counter()
    data = torch.load(data_path, map_location=device, weights_only=True)
    actor = Actor(sensor).to(device)
    critic = Critic().to(device)
    if initial:
        actor.load_state_dict(
            torch.load(initial, map_location=device, weights_only=True)["actor"]
        )
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4
    )
    n = len(data["x"])
    image_ids = data.get("image_id", torch.arange(n, device=device))
    label = torch.zeros(n, 4, 4, device=device)
    adv = torch.zeros(n, device=device)
    value = torch.zeros(n, device=device)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    vmask = mask.clone()
    shadow_manifest = None
    future = None
    if shadows:
        cache = torch.load(shadows, map_location=device, weights_only=True)
        shadow_manifest = cache["manifest"]
        if (
            shadow_manifest["dataset_sha256"]
            != hashlib.sha256(data_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Stale shadow cache: dataset hash differs")
        if shadow_manifest["sensor"] != sensor:
            raise ValueError("Shadow sensor mismatch")
        if shadow_manifest.get("physics") and json.dumps(
            shadow_manifest["physics"], sort_keys=True
        ) != json.dumps(asdict(cfg), sort_keys=True):
            raise ValueError("Stale shadow cache: dynamics/limits differ")
        if cache.get("future"):
            future = {
                key: torch.stack(
                    [
                        torch.as_tensor(row[key], device=device, dtype=torch.float32)
                        for row in cache["future"]
                    ]
                )
                for key in ["image", "x", "goal", "value"]
            }
        for row in cache["rows"]:
            j = row["index"]
            if not bool(data["train"][j]):
                raise ValueError("Shadow label leaks validation world")
            label[j] = row["knots"]
            adv[j] = row["advantage"]
            value[j] = row["value"]
            mask[j] = row["valid"]
            vmask[j] = True
    train_ids = data["train"].nonzero().flatten()
    valid_ids = (~data["train"]).nonzero().flatten()
    records = []
    best = float("inf")
    out.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def loss(ids):
        x = data["x"][ids]
        g = data["goal"][ids]
        b = data["boxes"][data["map_id"][ids]]
        knots = actor(data[sensor][image_ids[ids]], x, g)
        p, v, a, yaw = trajectory(x, knots)
        nav = navigation_cost(p, v, a, g, b, cfg).mean()
        # Feasibility cost for requested specific force, including tilt limit.
        force = a + a.new_tensor([0.0, 0.0, 9.81])
        tilt = torch.atan2(force[..., :2].norm(dim=-1), force[..., 2])
        feasible = (tilt - 0.6).clamp_min(0).square().mean() * 5 + (
            force.norm(dim=-1) - 19.62
        ).clamp_min(0).square().mean()
        feasible = (
            feasible + 2 * (a.norm(dim=-1) - cfg.max_accel).clamp_min(0).square().mean()
        )
        smooth = (knots[:, 1:] - knots[:, :-1]).square().mean()
        weight = mask[ids].float() * (1 + adv[ids].clamp(max=10))
        imitation = (
            (knots - label[ids]).square().mean((1, 2)) * weight
        ).sum() / weight.sum().clamp_min(1)
        # Current-state long-horizon return supervision. No map input to Critic.
        pred = critic(data[sensor][image_ids[ids]], x, g)
        vl = (
            F.smooth_l1_loss(pred, value[ids], reduction="none") * vmask[ids]
        ).sum() / vmask[ids].sum().clamp_min(1)
        if future is not None and actor.training:
            j = torch.randint(len(future["x"]), (min(8, len(ids)),), device=device)
            vl = vl + F.smooth_l1_loss(
                critic(future["image"][j], future["x"][j], future["goal"][j]),
                future["value"][j],
            )
        total = nav + feasible + 0.02 * smooth + 0.25 * imitation + 0.1 * vl
        collision = (clearance(p, b).amin(-1) < cfg.radius).float().mean()
        return total, torch.stack((nav, imitation, vl, collision))

    setup_s = time.perf_counter() - total_start
    for epoch in range(epochs):
        metrics = {}
        sync()
        begin = time.perf_counter()
        for name, ids in [
            ("train", train_ids[torch.randperm(len(train_ids), device=device)]),
            ("valid", valid_ids),
        ]:
            training = name == "train"
            actor.train(training)
            critic.train(training)
            sync()
            start = time.perf_counter()
            sums = torch.zeros(5, device=device)
            for selected in ids.split(batch_size):
                with torch.set_grad_enabled(training):
                    total, parts = loss(selected)
                    if not torch.isfinite(total):
                        raise RuntimeError("Nonfinite loss; checkpoint not accepted")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        total.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5)
                        optimizer.step()
                    sums += torch.cat(
                        (total.detach().reshape(1), parts.detach())
                    ) * len(selected)
            sync()
            elapsed = time.perf_counter() - start
            metrics[name] = dict(
                zip(
                    [
                        "total",
                        "navigation",
                        "imitation",
                        "value",
                        "predicted_collision_fraction",
                    ],
                    (sums / len(ids)).cpu().tolist(),
                )
            )
            metrics[name].update(seconds=elapsed, samples_per_s=len(ids) / elapsed)
        record = dict(epoch=epoch + 1, seconds=time.perf_counter() - begin, **metrics)
        records.append(record)
        print(json.dumps(record), flush=True)
        if metrics["valid"]["navigation"] < best:
            best = metrics["valid"]["navigation"]
            torch.save(
                dict(
                    actor=actor.state_dict(),
                    sensor=sensor,
                    physics=asdict(cfg),
                    epoch=epoch + 1,
                    algorithm="GAP-Nav continuous spline; not YOPO",
                    validation_navigation=best,
                    flight_validated=False,
                ),
                out / "best.pt",
            )
        torch.save(
            dict(
                actor=actor.state_dict(),
                sensor=sensor,
                physics=asdict(cfg),
                epoch=epoch + 1,
                flight_validated=False,
            ),
            out / "last.pt",
        )
        (out / "history.json").write_text(json.dumps(records, indent=2))
    torch.save(critic.state_dict(), out / "critic.pt")
    actor.eval()
    ids = valid_ids[:1]
    with torch.no_grad():
        for _ in range(10):
            actor(data[sensor][image_ids[ids]], data["x"][ids], data["goal"][ids])
        sync()
        t = time.perf_counter()
        for _ in range(100):
            actor(data[sensor][image_ids[ids]], data["x"][ids], data["goal"][ids])
        sync()
        latency = (time.perf_counter() - t) * 10
    manifest = dict(
        sensor=sensor,
        epochs=epochs,
        batch_size=batch_size,
        max_speed=max_speed,
        max_accel=max_accel,
        physics=asdict(cfg),
        parameters=sum(p.numel() for p in actor.parameters()),
        training_states=len(train_ids),
        unique_perception_views=len(data[sensor]),
        device=device,
        setup_s=setup_s,
        total_training_s=time.perf_counter() - total_start,
        steady_samples_per_s=sum(r["train"]["samples_per_s"] for r in records[1:])
        / max(1, len(records) - 1),
        inference_ms=latency,
        gpu_peak_mb=(
            torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
        ),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        shadow=shadow_manifest,
        checkpoint_sha256=hashlib.sha256((out / "best.pt").read_bytes()).hexdigest(),
        validation="held-out box worlds; predicted trajectory costs, NOT flight success",
        feature_version=2,
        initial_checkpoint=str(initial) if initial else None,
        flight_validated=False,
    )
    (out / "metrics.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sensor", choices=["depth", "mid360"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--shadows", type=Path)
    p.add_argument("--initial", type=Path)
    p.add_argument("--max-speed", type=float, default=6.0)
    p.add_argument("--max-accel", type=float, default=4.0)
    a = p.parse_args()
    train(
        a.data,
        a.output,
        a.sensor,
        a.epochs,
        a.batch_size,
        a.shadows,
        a.initial,
        a.max_speed,
        a.max_accel,
    )
