"""Calibrated optical Z depth, 20 m normalization, explicit unknown mask."""

import numpy as np


def preprocess(depth, height=96, width=160):
    import cv2

    depth = np.asarray(depth, dtype=np.float32)
    valid = (np.isfinite(depth) & (depth >= 0.2)) | np.isposinf(depth)
    meters = np.clip(np.nan_to_num(depth, nan=0, posinf=20, neginf=0), 0, 20)
    return np.stack(
        (
            cv2.resize(meters, (width, height), interpolation=cv2.INTER_NEAREST) / 20,
            cv2.resize(
                valid.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ),
        )
    )
