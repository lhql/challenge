#!/usr/bin/env python3
"""New camera poses in existing Astra box environments; offline batched rendering.

No YOPO planner/network/loss is imported. Whole environments are held out.
"""

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package))
from gap_nav.data import render_observation
from gap_nav.costs import clearance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    workspace = package.parents[1]
    sources = sorted((workspace / "dependencies/yopo400_dataset").glob("scene_*.npz"))
    if len(sources) < 4 or args.views % len(sources):
        raise ValueError("Need >=4 existing scenes and views divisible by scene count")
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    boxes = torch.as_tensor(
        np.stack([np.load(f)["boxes"] for f in sources]), device=device
    )
    ids = torch.arange(len(sources), device=device).repeat_interleave(
        args.views // len(sources)
    )
    x = torch.zeros(args.views, 13, device=device)
    pending = torch.ones(args.views, dtype=torch.bool, device=device)
    for _ in range(200):
        ix = pending.nonzero().flatten()
        if len(ix) == 0:
            break
        positions = torch.rand(len(ix), 3, device=device) * x.new_tensor(
            [16.0, 16.0, 2.6]
        ) + x.new_tensor([-8.0, -8.0, 0.65])
        valid = clearance(positions[:, None], boxes[ids[ix]]).squeeze(-1) > 0.65
        x[ix[valid], :3] = positions[valid]
        pending[ix[valid]] = False
    if pending.any():
        raise RuntimeError("Unable to sample free camera poses")
    attitude = np.column_stack(
        (
            rng.normal(0, 0.12, args.views),
            rng.normal(0, 0.15, args.views),
            rng.uniform(-np.pi, np.pi, args.views),
        )
    )
    q = Rotation.from_euler("xyz", attitude).as_quat()[:, [3, 0, 1, 2]]
    x[:, 6:10] = x.new_tensor(q)
    images = []
    for offset in range(0, len(x), 16):
        with torch.no_grad():
            images.append(
                render_observation(
                    x[offset : offset + 16],
                    boxes[ids[offset : offset + 16]],
                    "depth",
                    workspace,
                ).cpu()
            )
    data = dict(
        depth=torch.cat(images),
        x=x.cpu(),
        goal=x[:, :3].cpu(),
        boxes=boxes.cpu(),
        map_id=ids.cpu(),
        train=(ids < len(sources) - 2).cpu(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    report = dict(
        views=args.views,
        unique_worlds=len(sources),
        train_worlds=len(sources) - 2,
        held_out_worlds=2,
        seed=args.seed,
        generation_s=time.perf_counter() - start,
        camera_hfov_rad=1.5184364492350666,
        device=device,
        kind="new random camera poses in existing box environments; not original CUDA forest generator",
        sources=[
            dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sources
        ],
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
