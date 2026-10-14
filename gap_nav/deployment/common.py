"""ROS-free deployment helpers; no training or teacher dependencies."""

import json, os, time
from pathlib import Path
import numpy as np


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def vector(v):
    return np.array([v.x, v.y, v.z], dtype=float)


def evaluate(x, knots, t, duration=0.96):
    s = np.clip(t / duration, 0, 1)
    M = np.array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])
    c = M @ np.asarray(knots)
    vb = np.array([s ** (i + 1) / (i + 1) * duration for i in range(4)])
    pb = np.array([s ** (i + 2) / ((i + 1) * (i + 2)) * duration**2 for i in range(4)])
    a = np.array([s**i for i in range(4)]) @ c[:, :3]
    return (
        np.asarray(x[:3]) + np.asarray(x[3:6]) * t + pb @ c[:, :3],
        np.asarray(x[3:6]) + vb @ c[:, :3],
        a,
    )


def sitl_process(rootfs):
    """Only a locally running PX4 SITL in the launch-owned rootfs authorizes arming."""
    rootfs = Path(rootfs).resolve()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            exe = (proc / "exe").resolve()
            cwd = (proc / "cwd").resolve()
            if (
                cwd == rootfs
                and exe.name == "px4"
                and "px4_sitl" in str(exe)
                and b"-d" in (proc / "cmdline").read_bytes().split(b"\0")
            ):
                return True
        except (OSError, PermissionError):
            pass
    return False


def bounded_velocity(velocity, position, home, horizontal_limit=1.5):
    """Tracking envelope only: no obstacle map, search, or navigation decisions."""
    v = np.array(velocity, dtype=float, copy=True)
    horizontal = np.linalg.norm(v[:2])
    if horizontal > horizontal_limit:
        v[:2] *= horizontal_limit / horizontal
    # Match the existing PX4 vertical limits, with stopping room below ceiling.
    lower, upper = home[2] + 0.5, home[2] + 2.3
    v[2] = np.clip(
        v[2],
        np.clip(lower - position[2], -0.3, 0.5),
        np.clip(upper - position[2], -0.3, 0.5),
    )
    return v
