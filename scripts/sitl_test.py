#!/usr/bin/python3
"""Bounded SITL run; logs and kills only child process groups created here."""

import argparse, os, signal, subprocess, time, json, hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", choices=["depth", "mid360"], required=True)
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--weights", type=Path)
    p.add_argument("--name")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--world", type=Path)
    p.add_argument("--goal-x", type=float)
    p.add_argument("--goal-y", type=float, default=0.0)
    p.add_argument("--goal-z", type=float, default=0.0)
    a = p.parse_args()
    if a.goal_x is None:
        a.goal_x = 2.0 if a.demo else 4.0
    package = Path(__file__).resolve().parents[1]
    out = package / "results" / (a.name or ("sitl_" + a.sensor))
    out.mkdir(exist_ok=True)
    if (out / "events.jsonl").exists():
        raise SystemExit(
            "Results directory already contains a run; choose --name to preserve evidence"
        )
    with (out / "launch.log").open("w") as log:
        command = [
            str(package / "scripts/deploy.sh"),
            "--sensor",
            a.sensor,
            "auto_arm:=true",
        ]
        if a.demo:
            command = [
                str(package / "scripts/demo.sh"),
                "auto_arm:=true",
                "rviz:=false",
                "auto_goal:=false",
                "auto_land:=true",
            ]
        command.extend(
            [
                "gui:=" + str(a.gui).lower(),
                "goal_x:=" + str(a.goal_x),
                "goal_y:=" + str(a.goal_y),
                "goal_z:=" + str(a.goal_z),
            ]
        )
        world = (
            a.world.resolve()
            if a.world
            else package.parent
            / (
                "astra_yopo/worlds/jungle.world"
                if a.demo
                else "astra_yopo/worlds/yopo400.world"
            )
        )
        command.append("world:=" + str(world))
        weights = (
            a.weights
            if a.weights
            else package
            / "results"
            / ("depth_current" if a.sensor == "depth" else "mid360_dual")
            / "best.onnx"
        ).resolve()
        command.append("weights:=" + str(weights))
        (out / "run.json").write_text(
            json.dumps(
                dict(
                    sensor=a.sensor,
                    demo=a.demo,
                    weights=str(weights),
                    weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                    command=command,
                    seconds=a.seconds,
                    world=str(world),
                    goal_offset=[a.goal_x, a.goal_y, a.goal_z],
                ),
                indent=2,
            )
        )
        launch = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        observer_options = []
        if a.demo:
            observer_options = [
                "--demo-goal",
                str(a.goal_x),
                str(a.goal_y),
                str(a.goal_z),
            ]
        observer = subprocess.Popen(
            [
                "/usr/bin/python3",
                str(package / "scripts/observe_sitl.py"),
                "--output",
                str(out),
                "--seconds",
                str(a.seconds),
                *observer_options,
            ],
            start_new_session=True,
        )
        try:
            observer.wait(timeout=a.seconds + 10)
        finally:
            for proc in [observer, launch]:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGINT)
            for proc in [observer, launch]:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
        from analyze_flight import analyze

        if (out / "events.jsonl").exists():
            report = analyze(out)
            print(json.dumps(report, indent=2))
            if not report.get(
                "mission_completed" if a.demo else "flight_passed", False
            ):
                raise SystemExit(1)


if __name__ == "__main__":
    main()
