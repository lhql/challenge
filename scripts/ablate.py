#!/usr/bin/env python3
"""Matched-update ablations using the SAME recorded counterfactual branches.

This is not a claim of matched total wall-clock against the 2-epoch YOPO pilot.
All modes start from the same recorded base and receive equal continuation updates.
"""

import argparse, json, sys, time
from pathlib import Path
import torch

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package))
from gap_nav.learning.train import train
from gap_nav.trajectory import fit_knots, references
from gap_nav.dynamics import rollout, Physics
from gap_nav.costs import clearance, physical_cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", choices=["depth", "mid360"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--data", type=Path)
    p.add_argument("--base", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(2)
    root = package / "results"
    data_path = a.data or root / "data/samples.pt"
    base_path = a.base or root / f"{a.sensor}_base/best.pt"
    output_root = a.output or root / "ablations" / a.sensor
    data = torch.load(data_path, weights_only=True)
    original = torch.load(
        a.cache or root / f"{a.sensor}_shadows/shadows.pt", weights_only=True
    )
    cfg = Physics(**original["manifest"].get("physics", {}))
    results = []
    for mode in ["none", "mpc", "temporal", "policy", "dual"]:
        start = time.perf_counter()
        output = output_root / mode
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        if mode != "none":
            for record in original["details"]:
                j = record["index"]
                bb = data["boxes"][data["map_id"][j : j + 1]]
                xx = data["x"][j : j + 1]
                gg = data["goal"][j : j + 1]
                candidates = record["candidates"]
                temporal_count = len(record["temporal"])
                allowed = [0]
                if mode in ("temporal", "dual"):
                    allowed += list(range(1, temporal_count + 1))
                if mode in ("policy", "dual"):
                    allowed += list(range(temporal_count + 1, len(candidates)))
                chosen = min(
                    allowed,
                    key=lambda k: (
                        not bool(
                            (
                                clearance(candidates[k]["X"][..., :3], bb) > cfg.radius
                            ).all()
                        ),
                        float(candidates[k]["cost"][0]),
                    ),
                )
                c = candidates[chosen]
                knots = fit_knots(c["reference"][:, :12]).clamp(
                    -cfg.max_accel, cfg.max_accel
                )
                ref = torch.cat((references(xx, knots), c["reference"][:, 12:]), 1)
                with torch.no_grad():
                    X, _, _ = rollout(xx, ref, record["wind"], cfg=cfg)
                    cost = physical_cost(X, ref, gg, bb, cfg)
                valid = bool((clearance(X[..., :3], bb) > cfg.radius).all()) and bool(
                    cost <= record["baseline"]["cost"] + 1e-4
                )
                rows.append(
                    dict(
                        index=j,
                        knots=knots[0],
                        value=-cost[0],
                        advantage=(record["baseline"]["cost"] - cost).clamp_min(0)[0],
                        valid=valid,
                    )
                )
            manifest = dict(
                original["manifest"],
                mode=mode,
                cache_reselection_s=time.perf_counter() - start,
            )
            torch.save(
                dict(
                    rows=rows,
                    manifest=manifest,
                    future=(
                        original.get("future", [])
                        if mode in ("temporal", "dual")
                        else []
                    ),
                ),
                output / "shadows.pt",
            )
        metrics = train(
            data_path,
            output,
            a.sensor,
            a.epochs,
            a.batch_size,
            output / "shadows.pt" if mode != "none" else None,
            base_path,
            cfg.max_speed,
            cfg.max_accel,
        )
        metrics["ablation"] = mode
        results.append(metrics)
    (output_root / "summary.json").write_text(
        json.dumps(
            dict(
                initial_checkpoint=str(base_path),
                continuation_epochs=a.epochs,
                shared_shadow_generation_s=original["manifest"]["generation_s"],
                training_updates_matched=True,
                total_wall_clock_budget_matched=False,
                baseline_yopo_comparable=False,
                methods=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
