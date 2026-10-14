"""Reuse existing Astra YOPO box environments; cached GPU raycasting only offline."""

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .perception.mid360 import project, LIDAR_IN_BODY


def raycast(boxes, origin, R, rays, far=20.0):
    dirs = torch.einsum("bij,rj->bri", R, rays)
    inv = torch.where(dirs.abs() > 1e-9, 1 / dirs, torch.ones_like(dirs) * 1e9)
    distance = torch.full(dirs.shape[:2], far, device=dirs.device)
    for j in range(boxes.shape[1]):
        a = (boxes[:, j, 0, None] - origin[:, None]) * inv
        b = (boxes[:, j, 1, None] - origin[:, None]) * inv
        near = torch.minimum(a, b).amax(-1)
        end = torch.maximum(a, b).amin(-1)
        hit = (end >= near.clamp_min(0)) & (near >= 0)
        distance = torch.minimum(distance, torch.where(hit, near, far))
    return distance


def generate(ws, out, limit=1024, seed=42):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted((ws / "dependencies/yopo400_dataset").glob("scene_*.npz"))
    if len(files) < 4:
        raise RuntimeError("Existing Astra YOPO environment cache requires >=4 scenes")
    source = [dict(np.load(f)) for f in files]
    count = min(limit, sum(len(d["depth"]) for d in source))
    # Balanced worlds, entire last two maps held out (no shadow leakage).
    each = max(1, count // len(source))
    ids = np.repeat(np.arange(len(source)), each)
    pos = np.concatenate([d["position"][:each] for d in source])
    rot = np.concatenate([d["rotation"][:each] for d in source])
    count = len(pos)
    q = Rotation.from_matrix(rot).as_quat()[:, [3, 0, 1, 2]]
    vel_body = rng.normal([1.2, 0, 0], [1.1, 0.35, 0.2], (count, 3))
    vel_body[:, 0] = np.clip(vel_body[:, 0], 0, 4)
    vel_body[rng.random(count) < 0.2] = 0
    vel = np.einsum("bij,bj->bi", rot, vel_body)
    x = np.c_[pos, vel, q, rng.normal(0, 0.12, (count, 3))].astype(np.float32)
    goal_body = np.c_[
        rng.uniform(0.3, 8, count), rng.normal(0, 2, count), rng.normal(0, 0.3, count)
    ]
    goals = (pos + np.einsum("bij,bj->bi", rot, goal_body)).astype(np.float32)
    boxes = np.stack([d["boxes"] for d in source])
    # Camera exactly as existing d435 SDF: 87deg HFOV, 640x360 native.
    u, v = np.meshgrid(
        (np.arange(160) + 0.5) / 160 * 2 - 1, (np.arange(96) + 0.5) / 96 * 2 - 1
    )
    tangent = np.tan(1.5184364492350666 / 2)
    rays = np.stack(
        (np.ones_like(u), -u * tangent, -v * tangent * 360 / 640), -1
    ).reshape(-1, 3)
    scan_path = ws / "src/livox_laser_simulation_ros2/scan_mode/mid360.csv"
    scan = np.loadtxt(scan_path, delimiter=",", skiprows=1)
    depths = []
    lidars = []
    raw_points = []
    for offset in range(0, count, 8):
        sl = slice(offset, offset + 8)
        R = torch.tensor(rot[sl], device=device)
        p = torch.tensor(pos[sl], device=device)
        b = torch.tensor(boxes[ids[sl]], device=device)
        camera = p + R @ p.new_tensor([0.11, 0, 0.03])
        d = raycast(b, camera, R, torch.tensor(rays, dtype=p.dtype, device=device))
        depths.extend(
            torch.stack(
                (d.reshape(-1, 96, 160) / 20, torch.ones_like(d).reshape(-1, 96, 160)),
                1,
            )
            .cpu()
            .numpy()
        )
        # Match existing plugin's 20k-sample scan and downsample=4, phase varied.
        begin = int(rng.integers(0, len(scan)))
        rows = scan[(begin + np.arange(0, 20000, 4)) % len(scan)]
        az = np.deg2rad(rows[:, 1])
        el = np.pi / 2 - np.deg2rad(rows[:, 2])
        r = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1)
        sensor = p + R @ p.new_tensor(LIDAR_IN_BODY)
        ranges = (
            raycast(b, sensor, R, torch.tensor(r, dtype=p.dtype, device=device))
            .cpu()
            .numpy()
        )
        for distances in ranges:
            cloud = (
                r[distances < 19.99] * distances[distances < 19.99, None]
                + LIDAR_IN_BODY
            ).astype(np.float32)
            lidars.append(project(cloud))
            raw_points.append(cloud)
    output = dict(
        depth=torch.tensor(np.stack(depths)),
        mid360=torch.tensor(np.stack(lidars)),
        x=torch.tensor(x),
        goal=torch.tensor(goals),
        boxes=torch.tensor(boxes),
        map_id=torch.tensor(ids),
        train=torch.tensor(ids < len(files) - 2),
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.save(output, out / "samples.pt")
    # Ragged raw return cache, offsets preserve sample identity.
    np.savez_compressed(
        out / "mid360_returns.npz",
        points=np.concatenate(raw_points),
        offsets=np.cumsum([0] + [len(p) for p in raw_points]),
    )
    report = dict(
        samples=count,
        train=int(output["train"].sum()),
        valid=int((~output["train"]).sum()),
        generation_s=time.perf_counter() - start,
        device=device,
        seed=seed,
        environment="Existing Astra YOPO integration box-world cache",
        original_yopo_cuda=False,
        source=[
            {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
            for f in files
        ],
        scan_sha256=hashlib.sha256(scan_path.read_bytes()).hexdigest(),
        sensor_model="static pose scan, actual MID360 CSV pattern; live deskew implemented separately",
        limitations=[
            "Synthetic box worlds, not original tree mesh CUDA renderer",
            "No measured self-occlusion mask; body exclusion on point returns only",
        ],
    )
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=1024)
    a = p.parse_args()
    generate(a.workspace, a.output, a.samples)


def render_observation(x, boxes, sensor, workspace):
    """Offline-only observation at an inherited state, not a stale current image."""
    from .dynamics import rotation

    R = rotation(x[:, 6:10])
    p = x[:, :3]
    if sensor == "depth":
        u, v = torch.meshgrid(
            torch.arange(96, device=x.device),
            torch.arange(160, device=x.device),
            indexing="ij",
        )
        rays = (
            torch.stack(
                (
                    torch.ones_like(v),
                    (1 - 2 * (v + 0.5) / 160) * np.tan(1.5184364492350666 / 2),
                    (1 - 2 * (u + 0.5) / 96)
                    * np.tan(1.5184364492350666 / 2)
                    * 360
                    / 640,
                ),
                -1,
            )
            .reshape(-1, 3)
            .to(x.dtype)
        )
        d = raycast(boxes, p + R @ x.new_tensor([0.11, 0, 0.03]), R, rays).reshape(
            -1, 96, 160
        )
        return torch.stack((d / 20, torch.ones_like(d)), 1)
    scan = np.loadtxt(
        workspace / "src/livox_laser_simulation_ros2/scan_mode/mid360.csv",
        delimiter=",",
        skiprows=1,
        max_rows=20000,
    )[::4]
    az = np.deg2rad(scan[:, 1])
    el = np.pi / 2 - np.deg2rad(scan[:, 2])
    r = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1)
    d = (
        raycast(boxes, p + R @ x.new_tensor(LIDAR_IN_BODY), R, x.new_tensor(r))
        .cpu()
        .numpy()
    )
    images = [project(r[di < 19.99] * di[di < 19.99, None] + LIDAR_IN_BODY) for di in d]
    return x.new_tensor(np.stack(images))
