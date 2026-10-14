"""ENU/FLU rigid-body model. Quaternion order wxyz; actuator memory is cloneable."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Physics:
    mass: float = 1.586
    inertia: tuple = (0.02258842754, 0.0242628704, 0.04260194286)
    dt: float = 0.08
    substeps: int = 4
    response_s: float = 0.06
    max_accel: float = 4.0
    max_rate: float = 2.5
    max_tilt: float = 0.6
    max_speed: float = 6.0
    radius: float = (
        0.35  # Covers 0.327 m rotor tip radius plus vertical extent/padding.
    )


def multiply(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    v = (
        a[..., :1] * b[..., 1:]
        + b[..., :1] * a[..., 1:]
        + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    )
    return torch.cat((w, v), -1)


def rotation(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def initial_memory(x, cfg=Physics()):
    # Actual delayed [collective force N, body torque Nm].
    u = x.new_zeros((*x.shape[:-1], 4))
    u[..., 0] = cfg.mass * 9.81
    return u


def step(x, actuator, previous_command, reference, wind, cfg=Physics()):
    """One control interval, one-substep transport delay plus actuator lag.

    reference=[world acceleration xyz, yaw rate]; full physical forces/torques
    are calculated by the same tracking model for every counterfactual branch.
    """
    h = cfg.dt / cfg.substeps
    I = x.new_tensor(cfg.inertia)
    controls = []
    for _ in range(cfg.substeps):
        R = rotation(x[..., 6:10])
        acc = reference[..., :3].clamp(-cfg.max_accel, cfg.max_accel)
        gravity = acc.new_tensor([0.0, 0.0, 9.81])
        desired = acc + gravity
        z_des = desired / desired.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        z = R[..., :, 2]
        error_world = torch.cross(z, z_des, dim=-1)
        error_body = torch.matmul(
            R.transpose(-1, -2), error_world.unsqueeze(-1)
        ).squeeze(-1)
        rate_des = 5.0 * error_body + torch.stack(
            (reference[..., 3] * 0, reference[..., 3] * 0, reference[..., 3]), -1
        )
        rate_des = rate_des.clamp(-cfg.max_rate, cfg.max_rate)
        omega = x[..., 10:13]
        torque = (
            I * (rate_des - omega) / 0.12 + torch.cross(omega, I * omega, dim=-1)
        ).clamp(-0.35, 0.35)
        force = (cfg.mass * (desired * z).sum(-1)).clamp(
            0.2 * cfg.mass * 9.81, 2.0 * cfg.mass * 9.81
        )
        command = torch.cat((force.unsqueeze(-1), torque), -1)
        actuator = actuator + (previous_command - actuator) * (
            1.0 - torch.exp(x.new_tensor(-h / cfg.response_s))
        )
        previous_command = command
        acceleration = (
            z * (actuator[..., :1] / cfg.mass) - gravity - 0.12 * x[..., 3:6] + wind
        )
        omega_next = (
            omega + h * (actuator[..., 1:] - torch.cross(omega, I * omega, dim=-1)) / I
        )
        q = x[..., 6:10] + 0.5 * h * multiply(
            x[..., 6:10], torch.cat((omega[..., :1] * 0, (omega + omega_next) / 2), -1)
        )
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = x[..., 3:6] + h * acceleration
        p = x[..., :3] + h * (x[..., 3:6] + v) / 2
        x = torch.cat((p, v, q, omega_next), -1)
        controls.append(actuator)
    return x, actuator, previous_command, torch.stack(controls, -2)


def rollout(x, refs, wind=None, memory=None, cfg=Physics()):
    if memory is None:
        a = initial_memory(x, cfg)
        command = a.clone()
    else:
        a, command = memory
    if wind is None:
        wind = torch.zeros_like(refs[..., :3])
    states = [x]
    memories = [torch.stack((a, command), -2)]
    controls = []
    for k in range(refs.shape[-2]):
        x, a, command, u = step(x, a, command, refs[..., k, :], wind[..., k, :], cfg)
        states.append(x)
        memories.append(torch.stack((a, command), -2))
        controls.append(u)
    return torch.stack(states, -2), torch.stack(controls, -3), torch.stack(memories, -3)
