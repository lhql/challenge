import torch
from .dynamics import Physics, rotation


def clearance(points, boxes):
    c = boxes.mean(-2)
    half = (boxes[..., 1, :] - boxes[..., 0, :]) / 2
    d = (points[..., None, :] - c[:, None]).abs() - half[:, None]
    sdf = d.clamp_min(0).norm(dim=-1) + d.amax(-1).clamp_max(0)
    return sdf.amin(-1)


def navigation_cost(p, v, a, goal, boxes, cfg=Physics()):
    distance = clearance(p, boxes) - cfg.radius
    collision = torch.nn.functional.softplus((0.5 - distance) * 5).square().mean(-1) * 3
    guidance = (p[:, -1] - goal).norm(dim=-1) * 2
    close = (1 - (goal - p[:, 0]).norm(dim=-1) / 3).clamp(0, 1)
    braking = close * v[:, -1].square().sum(-1)
    speed = (v.norm(dim=-1) - cfg.max_speed).clamp_min(0).square().mean(-1)
    smooth = (a[:, 1:] - a[:, :-1]).square().mean((1, 2))
    effort = a.square().mean((1, 2)) * 0.025
    return collision + guidance + braking + speed + 0.03 * smooth + effort


def physical_cost(X, refs, goal, boxes, cfg=Physics()):
    a = (X[:, 1:, 3:6] - X[:, :-1, 3:6]) / cfg.dt
    base = navigation_cost(X[:, 1:, :3], X[:, 1:, 3:6], a, goal, boxes, cfg)
    R = rotation(X[..., 6:10])
    tilt = torch.acos(R[..., 2, 2].clamp(-0.999999, 0.999999))
    constraints = 8 * (tilt - cfg.max_tilt).clamp_min(0).square().mean(-1) + 2 * (
        X[..., 10:].norm(dim=-1) - cfg.max_rate
    ).clamp_min(0).square().mean(-1)
    # Remaining stopping distance penalizes unavoidably fast terminal states.
    margin = (
        clearance(X[:, -1:, :3], boxes).squeeze(-1)
        - cfg.radius
        - X[:, -1, 3:6].square().sum(-1) / (2 * cfg.max_accel)
    )
    return base + constraints + 2 * (-margin).clamp_min(0).square()
