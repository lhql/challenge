#!/usr/bin/env python3
"""Run unmodified upstream YOPO trainer on the existing Astra environment cache."""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    package = Path(__file__).resolve().parents[1]
    ws = package.parents[1]
    out = package / "results/baseline"
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [
        str(ws / "dependencies/yopo_python"),
        str(ws / "src/YOPO-ROS2/YOPO"),
    ]
    sys.path.append(str(package / ".deps"))
    import cv2, open3d as o3d
    from scipy.spatial.transform import Rotation
    from ruamel.yaml import YAML

    start = time.perf_counter()
    dataset = out / "dataset"
    dataset.mkdir(exist_ok=True)
    for i, f in enumerate(
        sorted((ws / "dependencies/yopo400_dataset").glob("scene_*.npz"))
    ):
        d = np.load(f)
        folder = dataset / str(i)
        folder.mkdir(exist_ok=True)
        for j, image in enumerate(d["depth"]):
            cv2.imwrite(
                str(folder / f"depth_{j}.png"),
                (image.astype(float) * 65535).astype(np.uint16),
            )
        q = Rotation.from_matrix(d["rotation"]).as_quat()[:, [3, 0, 1, 2]]
        np.savetxt(
            dataset / f"pose-{i}.csv",
            np.c_[d["position"], q],
            delimiter=",",
            header="x,y,z,qw,qx,qy,qz",
            comments="",
        )
        points = []
        for lo, hi in d["boxes"]:
            for axis in range(3):
                other = [k for k in range(3) if k != axis]
                grid = (
                    np.array(
                        np.meshgrid(
                            *[np.arange(lo[k], hi[k] + 0.1, 0.15) for k in other]
                        )
                    )
                    .reshape(2, -1)
                    .T
                )
                for boundary in [lo[axis], hi[axis]]:
                    xyz = np.zeros((len(grid), 3))
                    xyz[:, other] = grid
                    xyz[:, axis] = boundary
                    points.append(xyz)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.concatenate(points))
        o3d.io.write_point_cloud(str(dataset / f"pointcloud-{i}.ply"), pc)
    conversion = time.perf_counter() - start
    cfg = YAML().load((ws / "src/YOPO-ROS2/YOPO/config/traj_opt.yaml").read_text())
    cfg["dataset_path"] = str(dataset)
    config = out / "config.yaml"
    with config.open("w") as f:
        YAML().dump(cfg, f)
    os.environ["YOPO_CONFIG"] = str(config)
    from policy.yopo_trainer import YopoTrainer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer = YopoTrainer(
        batch_size=a.batch_size,
        loss_weight=[1.0, 1.0],
        tensorboard_path=str(out),
        checkpoint_path=str(ws / "src/YOPO-ROS2/YOPO/saved/YOPO_1/epoch50.pth"),
    )
    setup = time.perf_counter() - start
    original = trainer.train_one_epoch
    records = []

    def measured(*args):
        torch.cuda.synchronize()
        t = time.perf_counter()
        original(*args)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        n = (len(trainer.train_dataloader.dataset) // a.batch_size) * a.batch_size
        records.append({"seconds": seconds, "samples": n, "samples_per_s": n / seconds})

    trainer.train_one_epoch = measured
    start = time.perf_counter()
    trainer.train(a.epochs, save_interval=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    trainer.policy.eval()
    image = torch.ones(1, 1, 96, 160, device="cuda")
    obs = torch.zeros(1, 9, device="cuda")
    with torch.no_grad():
        for _ in range(10):
            trainer.policy.inference(image, obs.clone())
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100):
            trainer.policy.inference(image, obs.clone())
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t) * 10
    report = dict(
        kind="upstream YOPO network, loss and trainer; converted existing Astra box-world cache",
        original_cuda_generator="not built: CMake requires CUDA 12.8 EXACT; installed toolkit 12.3",
        batch_size=a.batch_size,
        epochs=records,
        training_and_validation_s=elapsed,
        cache_conversion_s=conversion,
        setup_esdf_s=setup,
        total_measured_s=conversion + setup + elapsed,
        inference_ms=latency,
        gpu_peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
        gpu_peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
        source_data_generation_s=None,
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
