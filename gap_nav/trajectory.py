"""Independent continuous cubic acceleration spline, integrated analytically."""

import torch


def trajectory(x, knots, duration=0.96, samples=25):
    t = torch.linspace(0, 1, samples, device=x.device, dtype=x.dtype)
    # Cubic Bezier acceleration; no discrete primitives or YOPO policy imports.
    M = x.new_tensor([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])
    co = torch.einsum("ij,bjk->bik", M, knots)
    basis = torch.stack([t**i for i in range(4)], -1)
    vb = torch.stack([t ** (i + 1) / (i + 1) * duration for i in range(4)], -1)
    pb = torch.stack(
        [t ** (i + 2) / ((i + 1) * (i + 2)) * duration**2 for i in range(4)], -1
    )
    a = torch.einsum("ti,bik->btk", basis, co[..., :3])
    v = x[:, None, 3:6] + torch.einsum("ti,bik->btk", vb, co[..., :3])
    p = (
        x[:, None, :3]
        + x[:, None, 3:6] * t[None, :, None] * duration
        + torch.einsum("ti,bik->btk", pb, co[..., :3])
    )
    yaw_rate = torch.einsum("ti,bi->bt", basis, co[..., 3])
    return p, v, a, yaw_rate


def references(x, knots, n=12):
    _, _, a, y = trajectory(x, knots, samples=n + 1)
    return torch.cat((a[:, :-1], y[:, :-1, None]), -1)


def fit_knots(refs):
    t = torch.linspace(0, 1, refs.shape[-2] + 1, device=refs.device, dtype=refs.dtype)[
        :-1
    ]
    B = torch.stack(((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3), -1)
    return torch.einsum("kn,bnd->bkd", torch.linalg.pinv(B), refs)
