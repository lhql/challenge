import torch
from ..dynamics import rollout
from ..costs import physical_cost


def challenge(teacher, baseline, actor_reference, goal, boxes, wind):
    """Two structured continuous candidates; identical state, wind and duration."""
    base = baseline["reference"]
    n = base.shape[1]
    actor = torch.cat(
        (actor_reference, torch.zeros_like(base[:, actor_reference.shape[1] :])), 1
    )
    brake = base.clone()
    brake[:, : n // 2, :3] *= 0.7
    brake[:, n // 2 :, :3] = (-baseline["X"][:, n // 2 : n, 3:6] * 1.2).clamp(-4, 4)
    candidates = []
    for ref in [actor, brake]:
        with torch.no_grad():
            X, U, M = rollout(baseline["X"][:, 0], ref, wind, cfg=teacher.cfg)
            cost = physical_cost(X, ref, goal, boxes, teacher.cfg)
        candidates.append(
            dict(
                X=X,
                U=U,
                memory=M,
                reference=ref,
                cost=cost,
                success=torch.isfinite(cost),
            )
        )
    return candidates
