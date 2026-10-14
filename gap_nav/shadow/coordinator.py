"""Sparse offline dual shadows, GPU-batched by inherited prefix length."""

import argparse, hashlib, json, time
from dataclasses import asdict
from pathlib import Path
import torch
from ..mpc.teacher import Teacher
from ..dynamics import rollout, Physics
from ..costs import clearance, physical_cost
from ..trajectory import references, fit_knots
from ..learning.networks import Actor
from ..data import render_observation
from .temporal import keyframes, inherit
from .policy import challenge


def cpu(o):
    if torch.is_tensor(o):
        return o.detach().cpu()
    if isinstance(o, dict):
        return {k: cpu(v) for k, v in o.items()}
    if isinstance(o, list):
        return [cpu(v) for v in o]
    return o


def subset(solution, ids):
    return {
        k: v[ids] if torch.is_tensor(v) and v.ndim else v for k, v in solution.items()
    }


def generate(
    data_path,
    output,
    actor_path=None,
    sensor="depth",
    fraction=0.03,
    mode="dual",
    iterations=12,
    max_speed=6.0,
    max_accel=4.0,
    max_batch=64,
):
    if not 0 < fraction <= 0.05:
        raise ValueError("Sparse fraction must be in (0, .05]")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(data_path, map_location=device, weights_only=True)
    actor = Actor(sensor).to(device)
    if actor_path:
        actor.load_state_dict(
            torch.load(actor_path, map_location=device, weights_only=True)["actor"]
        )
    elif mode in ("dual", "policy"):
        raise ValueError("Policy shadows require a trained base Actor")
    actor.eval()
    x = data["x"]
    boxes = data["boxes"][data["map_id"]]
    goal = data["goal"]
    images = data.get("image_id", torch.arange(len(x), device=device))
    score = x[:, 3:6].norm(dim=-1).square() / clearance(x[:, None, :3], boxes).squeeze(
        -1
    ).clamp_min(0.1)
    score[~data["train"]] = -torch.inf
    count = max(1, int(int(data["train"].sum()) * fraction))
    selected = score.topk(count).indices
    teacher = Teacher(
        cfg=Physics(max_speed=max_speed, max_accel=max_accel), iterations=iterations
    )
    rows = []
    details = []
    future = []
    workspace = Path(__file__).resolve().parents[4]
    # Bound graph and perception memory on a 4 GB GPU. No MPC in gradient batches.
    for chosen_ids in selected.split(max_batch):
        xx = x[chosen_ids]
        bb = boxes[chosen_ids]
        gg = goal[chosen_ids]
        n = len(chosen_ids)
        wind = torch.randn(n, 24, 3, device=device) * 0.06
        baseline = teacher.solve(xx, gg, bb, wind=wind[:, :12])
        extension = torch.zeros(n, 12, 4, device=device)
        extension[..., :3] = (-baseline["X"][:, -1:, 3:6]).clamp(-max_accel, max_accel)
        ref = torch.cat((baseline["reference"], extension), 1)
        with torch.no_grad():
            X, U, M = rollout(xx, ref, wind, cfg=teacher.cfg)
            cost = physical_cost(X, ref, gg, bb, teacher.cfg)
        base = dict(
            X=X, U=U, memory=M, reference=ref, cost=cost, success=torch.isfinite(cost)
        )
        candidates = [[subset(base, slice(j, j + 1))] for j in range(n)]
        metadata = [[] for _ in range(n)]
        if mode in ("temporal", "dual"):
            keys = keyframes(baseline, bb, cfg=teacher.cfg)
            for rank in range(keys.shape[1]):
                for k in keys[:, rank].unique().tolist():
                    members = (keys[:, rank] == k).nonzero().flatten()
                    solution, info = inherit(
                        teacher,
                        subset(base, members),
                        gg[members],
                        bb[members],
                        wind[members],
                        int(k),
                    )
                    with torch.no_grad():
                        observations = render_observation(
                            info["state"].to(device), bb[members], sensor, workspace
                        ).cpu()
                    for local, j in enumerate(members.tolist()):
                        candidates[j].append(subset(solution, slice(local, local + 1)))
                        record = dict(
                            k=int(k),
                            clone_error=info["clone_error"],
                            return_to_go=float(info["returns_to_go"][local]),
                            state=info["state"][local : local + 1],
                            memory=info["memory"][local : local + 1],
                            future_only_first_unchanged=True,
                            prefix_changed=bool(
                                not torch.equal(
                                    solution["reference"][local, 0], ref[j, 0]
                                )
                            ),
                        )
                        metadata[j].append(record)
                        future.append(
                            dict(
                                image=observations[local],
                                x=record["state"][0],
                                goal=gg[j].cpu(),
                                value=record["return_to_go"],
                            )
                        )
        if mode in ("policy", "dual"):
            with torch.no_grad():
                act = references(xx, actor(data[sensor][images[chosen_ids]], xx, gg))
            for candidate in challenge(teacher, base, act, gg, bb, wind):
                for j in range(n):
                    candidates[j].append(subset(candidate, slice(j, j + 1)))
        for j, index in enumerate(chosen_ids.tolist()):
            cc = candidates[j]
            b = bb[j : j + 1]
            g = gg[j : j + 1]
            state = xx[j : j + 1]
            feasible = [
                bool((clearance(c["X"][..., :3], b) > teacher.cfg.radius).all())
                for c in cc
            ]
            winner = min(
                range(len(cc)), key=lambda k: (not feasible[k], float(cc[k]["cost"][0]))
            )
            picked = cc[winner]
            knots = fit_knots(picked["reference"][:, :12]).clamp(-max_accel, max_accel)
            fitted = torch.cat(
                (references(state, knots), picked["reference"][:, 12:]), 1
            )
            with torch.no_grad():
                fx, _, _ = rollout(state, fitted, wind[j : j + 1], cfg=teacher.cfg)
                fit_cost = physical_cost(fx, fitted, g, b, teacher.cfg)
            valid = bool(
                (clearance(fx[..., :3], b) > teacher.cfg.radius).all()
            ) and bool(fit_cost <= base["cost"][j] + 1e-4)
            rows.append(
                dict(
                    index=index,
                    knots=knots.cpu()[0],
                    value=-picked["cost"].cpu()[0],
                    advantage=(base["cost"][j] - fit_cost).clamp_min(0).cpu()[0],
                    valid=valid,
                )
            )
            details.append(
                dict(
                    index=index,
                    mpc=subset(baseline, slice(j, j + 1)),
                    baseline=cc[0],
                    winner=winner,
                    candidates=cc,
                    temporal=metadata[j],
                    wind=wind[j : j + 1],
                    fit_cost=fit_cost,
                    projection_accepted=valid,
                )
            )
        print(
            json.dumps(
                dict(
                    completed=len(rows),
                    selected=count,
                    accepted=sum(r["valid"] for r in rows),
                    elapsed_s=time.perf_counter() - start,
                )
            ),
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        mode=mode,
        sensor=sensor,
        max_speed=max_speed,
        max_accel=max_accel,
        physics=asdict(teacher.cfg),
        fraction=fraction,
        selected=count,
        accepted=sum(r["valid"] for r in rows),
        generation_s=time.perf_counter() - start,
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        actor_sha256=(
            hashlib.sha256(actor_path.read_bytes()).hexdigest() if actor_path else None
        ),
        seed=42,
        iterations=iterations,
        solver="batched by temporal prefix length",
        max_batch=max_batch,
    )
    torch.save(
        dict(rows=cpu(rows), details=cpu(details), future=cpu(future), manifest=report),
        output / "shadows.pt",
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--actor", type=Path)
    p.add_argument("--sensor", choices=["depth", "mid360"], default="depth")
    p.add_argument("--fraction", type=float, default=0.03)
    p.add_argument(
        "--mode", choices=["mpc", "temporal", "policy", "dual"], default="dual"
    )
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--max-speed", type=float, default=6.0)
    p.add_argument("--max-accel", type=float, default=4.0)
    p.add_argument("--max-batch", type=int, default=64)
    a = p.parse_args()
    generate(
        a.data,
        a.output,
        a.actor,
        a.sensor,
        a.fraction,
        a.mode,
        a.iterations,
        a.max_speed,
        a.max_accel,
        a.max_batch,
    )
