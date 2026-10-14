"""Sparse differentiable shooting MPC; stores ALL states/controls/actuator memory."""

import time
import torch
from ..dynamics import Physics, rollout
from ..costs import physical_cost


class Teacher:
    def __init__(self, cfg=Physics(), iterations=18):
        self.cfg = cfg
        self.iterations = iterations

    def solve(
        self, x, goal, boxes, n=12, warm=None, memory=None, wind=None, freeze_after=None
    ):
        if x.ndim != 2 or x.shape[-1] != 13 or not torch.isfinite(x).all():
            raise ValueError("MPC needs finite [batch,13] states")
        if n < 2 or not torch.isfinite(goal).all() or not torch.isfinite(boxes).all():
            raise ValueError("Invalid MPC horizon, goal or environment")
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        if warm is None:
            target = (goal - x[:, :3]) * 0.8 - x[:, 3:6] * 1.5
            warm = torch.cat((target.clamp(-3, 3), x[:, :1] * 0), -1)[:, None].repeat(
                1, n, 1
            )
        if warm.shape != (len(x), n, 4) or not torch.isfinite(warm).all():
            raise ValueError("Invalid warm start")
        # Project the initial incumbent too: a saturated rollout must not retain
        # an out-of-bounds reference merely because its cost equals the bound.
        warm = torch.cat(
            (
                warm[..., :3].clamp(-self.cfg.max_accel, self.cfg.max_accel),
                warm[..., 3:].clamp(-1.5, 1.5),
            ),
            -1,
        )
        failed = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        u = warm.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=0.25)
        best_u = warm.detach().clone()
        with torch.no_grad():
            X, _, _ = rollout(x, best_u, wind, memory, self.cfg)
            best = physical_cost(X, best_u, goal, boxes, self.cfg)
        for _ in range(self.iterations):
            opt.zero_grad()
            X, _, _ = rollout(x, u, wind, memory, self.cfg)
            costs = physical_cost(X, u, goal, boxes, self.cfg)
            costs.sum().backward()
            if not torch.isfinite(u.grad).all():
                failed |= ~torch.isfinite(u.grad).all(dim=(1, 2))
                break
            if freeze_after is not None:
                u.grad[:, freeze_after:] = 0
            opt.step()
            with torch.no_grad():
                u[..., :3].clamp_(-self.cfg.max_accel, self.cfg.max_accel)
                u[..., 3].clamp_(-1.5, 1.5)
                if freeze_after is not None:
                    u[:, freeze_after:] = warm[:, freeze_after:]
                X, _, _ = rollout(x, u, wind, memory, self.cfg)
                c = physical_cost(X, u, goal, boxes, self.cfg)
                improved = torch.isfinite(c) & (c < best)
                best = torch.where(improved, c, best)
                best_u = torch.where(improved[:, None, None], u.detach(), best_u)
        with torch.no_grad():
            X, U, M = rollout(x, best_u, wind, memory, self.cfg)
        if x.is_cuda:
            torch.cuda.synchronize()
        return dict(
            X=X.detach(),
            U=U.detach(),
            memory=M.detach(),
            reference=best_u.detach(),
            cost=best.detach(),
            solve_s=time.perf_counter() - start,
            success=torch.isfinite(best) & ~failed,
            gradient_failure=failed,
        )
