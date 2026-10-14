#!/usr/bin/env python3
"""Compute measured flight metrics, with conservative geometry clearance labels."""

import argparse, json, math, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def boxes_from_world(path):
    boxes = []
    for model in ET.parse(path).findall(".//world/model"):
        if model.findtext("static") != "true":
            continue
        pose = np.fromstring(model.findtext("pose", "0 0 0 0 0 0"), sep=" ")
        # Existing test world uses axis-aligned boxes; do not silently approximate rotations.
        if np.linalg.norm(pose[3:]) > 1e-8:
            continue
        for link in model.findall("link"):
            lp = np.fromstring(link.findtext("pose", "0 0 0 0 0 0"), sep=" ")
            if np.linalg.norm(lp[3:]) > 1e-8:
                continue
            for collision in link.findall("collision"):
                size = collision.findtext("geometry/box/size")
                if size is None:
                    continue
                cp = np.fromstring(collision.findtext("pose", "0 0 0 0 0 0"), sep=" ")
                if np.linalg.norm(cp[3:]) > 1e-8:
                    continue
                center = pose[:3] + lp[:3] + cp[:3]
                half = np.fromstring(size, sep=" ") / 2
                boxes.append([center - half, center + half])
    return np.array(boxes)


def analyze(directory):
    rows = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]
    truth = [r for r in rows if r["type"] == "truth"]
    control = [r for r in rows if r["type"] == "control_status"]
    fcu = [r for r in rows if r["type"] == "fcu"]
    phase_order = []
    for r in control:
        if not phase_order or r["phase"] != phase_order[-1]:
            phase_order.append(r["phase"])
    if not truth:
        return dict(
            run=str(directory), flight_passed=False, reason="No ground-truth recording"
        )
    p = np.array([r["p"] for r in truth])
    v = np.array([r["v"] for r in truth])
    t = np.array([r["stamp"] for r in truth])
    q = np.array([r["q"] for r in truth])
    wall = np.array([r["wall"] for r in truth])
    armed = [r["wall"] for r in fcu if r["armed"]]
    start = min(armed) if armed else float("inf")
    airborne = (wall >= start) & (p[:, 2] > 0.2)
    angles = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_euler("xyz")
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    dt = np.maximum(np.diff(t), 1e-4)
    rate = (rotations[:-1].inv() * rotations[1:]).magnitude() / dt
    acceleration = np.diff(v, axis=0) / dt[:, None]
    jerk = (
        np.diff(acceleration, axis=0)
        / np.maximum((dt[1:] + dt[:-1]) / 2, 1e-4)[:, None]
    )
    run = (
        json.loads((directory / "run.json").read_text())
        if (directory / "run.json").exists()
        else {}
    )
    world = Path(
        run.get("world", directory.parents[2] / "astra_yopo/worlds/yopo400.world")
    )
    if not world.exists():
        world = Path(__file__).resolve().parents[2] / "astra_yopo/worlds/yopo400.world"
    boxes = boxes_from_world(world)
    distance = None
    if len(boxes) and airborne.any():
        delta = (
            np.abs(p[airborne, None] - boxes.mean(1)) - (boxes[:, 1] - boxes[:, 0]) / 2
        )
        distance = (
            np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(-1), 0)
        ).min(-1) - 0.35
    goal_reached = any(
        r.get(
            "goal_reached", r.get("reason") == "Goal reached" or r["phase"] == "HOVER"
        )
        for r in control
    )
    ended_disarmed = bool(fcu and not fcu[-1]["armed"])
    clearance_violation = bool(distance is not None and distance.min() <= 0)
    health = [
        r["inference_ms"]
        for r in rows
        if r["type"] == "inference_health" and r.get("ready")
    ]
    flight_time = float(np.sum(np.diff(t) * airborne[:-1]))
    report = dict(
        run=directory.name,
        checkpoint=run.get("weights"),
        checkpoint_sha256=run.get("weights_sha256"),
        phases=phase_order,
        goal_reached=goal_reached,
        mission_completed=bool(goal_reached and ended_disarmed),
        clearance_geometry_available=distance is not None,
        ended_disarmed=ended_disarmed,
        flight_passed=bool(
            goal_reached
            and ended_disarmed
            and distance is not None
            and not clearance_violation
        ),
        airborne_s=flight_time,
        mean_speed_m_s=(
            float(np.linalg.norm(v[airborne], axis=-1).mean())
            if airborne.any()
            else None
        ),
        max_speed_m_s=float(np.linalg.norm(v, axis=-1).max()),
        max_altitude_m=float(p[:, 2].max()),
        min_conservative_obstacle_clearance_m=(
            float(distance.min()) if distance is not None else None
        ),
        clearance_violation=clearance_violation,
        collision_sensor_available=False,
        collision_rate=None,
        max_roll_pitch_rad=(
            np.abs(angles[airborne, :2]).max(0).tolist() if airborne.any() else None
        ),
        max_angular_rate_rad_s=(
            float(rate[airborne[:-1]].max()) if airborne[:-1].any() else None
        ),
        mean_jerk_m_s3=(
            float(np.linalg.norm(jerk[airborne[:-2]], axis=-1).mean())
            if airborne[:-2].any()
            else None
        ),
        mean_sensor_preprocess_plus_inference_ms=(
            float(np.mean(health)) if health else None
        ),
        terminal_reason=control[-1].get("reason") if control else None,
        note="Actual PX4/Gazebo truth and ROS messages. Clearance is a conservative 0.35m sphere (0.327m rotor envelope plus padding) vs axis-aligned world boxes, not a contact sensor. No inferred collision count is presented as measured collision rate.",
    )
    (directory / "analysis.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps(analyze(directory)))
