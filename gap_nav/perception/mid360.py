"""Unordered MID360 returns -> conservative 360-degree polar minimum/mask.

Missing cells remain unknown. Not a spinning-lidar range image. No multi-frame
accumulation is enabled by default. Point times must be compensated first.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

LIDAR_IN_BODY = np.array([0.0, 0.0, 0.13])


def project(points, height=32, width=128, far=20.0):
    p = np.asarray(points, dtype=np.float32)
    p = p[np.isfinite(p).all(1)]
    # Body hull and mount exclusion in FLU, before polar representation.
    hull = (
        (np.abs(p[:, 0]) < 0.23)
        & (np.abs(p[:, 1]) < 0.23)
        & (p[:, 2] > -0.12)
        & (p[:, 2] < 0.23)
    )
    d = np.linalg.norm(p, axis=1)
    valid = (d >= 0.3) & (d < far) & ~hull
    p = p[valid]
    d = d[valid]
    az = np.arctan2(p[:, 1], p[:, 0])
    el = np.arcsin(np.clip(p[:, 2] / d, -1, 1))
    v = np.floor((el + np.pi / 6) / (2 * np.pi / 3) * height).astype(int)
    u = np.floor((az + np.pi) / (2 * np.pi) * width).astype(int) % width
    keep = (v >= 0) & (v < height)
    ids = v[keep] * width + u[keep]
    result = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(result, ids, d[keep])
    mask = np.isfinite(result)
    result = np.where(mask, result / far, 0)
    return np.stack(
        (result.reshape(height, width), mask.reshape(height, width).astype(np.float32))
    )


def deskew(points, times, pose_times, positions, quaternions_xyzw, reference_time):
    """Per-point sensor -> world -> current body; refuses missing pose coverage."""
    pt = np.asarray(pose_times)
    times = np.asarray(times)
    if (
        len(pt) < 2
        or times.min() < pt[0]
        or max(times.max(), reference_time) > pt[-1]
        or reference_time < pt[0]
    ):
        raise ValueError(
            "Pose history does not cover scan; dropping, never silently extrapolating"
        )
    rot = Slerp(pt, Rotation.from_quat(quaternions_xyzw))
    r = rot(times)
    rr = rot([reference_time])[0]
    position = np.stack(
        [np.interp(times, pt, np.asarray(positions)[:, k]) for k in range(3)], -1
    )
    ref = np.array(
        [np.interp(reference_time, pt, np.asarray(positions)[:, k]) for k in range(3)]
    )
    return rr.inv().apply(r.apply(np.asarray(points) + LIDAR_IN_BODY) + position - ref)
