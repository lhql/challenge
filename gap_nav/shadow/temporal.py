"""Risk-ranked inheritance with exact actuator state and prefix re-optimization."""

import torch
from ..dynamics import rotation, rollout, Physics
from ..costs import clearance, physical_cost


def keyframes(solution, boxes, count=2, cfg=Physics()):
    X = solution["X"]
    R = rotation(X[..., 6:10])
    speed = X[..., 3:6].norm(dim=-1)
    gap = clearance(X[..., :3], boxes) - cfg.radius
    risk = (
        speed.square() / (2 * cfg.max_accel * (gap.clamp_min(0.1)))
        + 2 * (1 - R[..., 2, 2])
        + X[..., 10:].norm(dim=-1) * 0.2
    )
    risk[:, 0] = -torch.inf
    risk[:, -1] = -torch.inf
    return risk.topk(min(count, X.shape[1] - 2), dim=1).indices


def inherit(teacher, baseline, goal, boxes, wind, k):
    """A batch of cases sharing the inherited index. Full state clone includes lag and delayed command.

    Optimize suffix at k, then re-optimize ONLY prefix from the original state.
    Any improved present-action target comes from this second optimization.
    """
    X = baseline["X"]
    M = baseline["memory"]
    ref = baseline["reference"]
    memory = (M[:, k, 0], M[:, k, 1])
    suffix = teacher.solve(
        X[:, k],
        goal,
        boxes,
        n=ref.shape[1] - k,
        warm=ref[:, k:],
        memory=memory,
        wind=wind[:, k:],
    )
    inherited = torch.cat((ref[:, :k], suffix["reference"]), 1)
    with torch.no_grad():
        joined, _, _ = rollout(X[:, 0], inherited, wind, cfg=teacher.cfg)
        error = (joined[:, k] - X[:, k]).abs().max().item()
    if error > 1e-5:
        raise RuntimeError("Dynamics continuity violation at inherited state")
    improved = teacher.solve(
        X[:, 0], goal, boxes, n=ref.shape[1], warm=inherited, wind=wind, freeze_after=k
    )
    return improved, dict(
        k=k,
        clone_error=error,
        return_to_go=-float(suffix["cost"][0]),
        returns_to_go=-suffix["cost"].cpu(),
        state=X[:, k].cpu(),
        memory=M[:, k].cpu(),
        future_only_first_unchanged=bool(torch.equal(inherited[:, 0], ref[:, 0])),
        prefix_changed=bool(not torch.equal(improved["reference"][:, 0], ref[:, 0])),
    )
