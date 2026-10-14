#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package))
sys.path.append(str(package / ".deps"))
import numpy as np
import torch
from gap_nav.learning.networks import Actor


class ExactPool(torch.nn.Module):
    """Static spatial bins, identical to adaptive average pooling; dynamic batch."""

    def forward(self, x):
        import math

        h, w = x.shape[-2:]
        return torch.cat(
            [
                torch.cat(
                    [
                        x[
                            :,
                            :,
                            math.floor(i * h / 3) : math.ceil((i + 1) * h / 3),
                            math.floor(j * w / 5) : math.ceil((j + 1) * w / 5),
                        ].mean((-2, -1), keepdim=True)
                        for j in range(5)
                    ],
                    -1,
                )
                for i in range(3)
            ],
            -2,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    actor = Actor(checkpoint["sensor"])
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    size = (96, 160) if checkpoint["sensor"] == "depth" else (32, 128)
    image = torch.rand(1, 2, *size)
    x = torch.zeros(1, 13)
    x[:, 6] = 1
    goal = torch.tensor([[3.0, 1.0, 0.0]])
    with torch.no_grad():
        before = actor(image, x, goal)
        actor.encoder.net[6] = ExactPool()
        assert torch.allclose(before, actor(image, x, goal), atol=1e-5)
    path = a.checkpoint.with_suffix(".onnx")
    torch.onnx.export(
        actor,
        (image, x, goal),
        str(path),
        input_names=["perception", "state", "goal"],
        output_names=["acceleration_yaw_knots"],
        dynamic_axes={
            name: {0: "batch"}
            for name in ["perception", "state", "goal", "acceleration_yaw_knots"]
        },
        opset_version=17,
        dynamo=False,
    )
    import onnx, onnxruntime as ort

    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    errors = []
    for seed in range(10):
        torch.manual_seed(seed)
        image = torch.rand(2, 2, *size)
        x = torch.randn(2, 13)
        x[:, 6:10] /= x[:, 6:10].norm(dim=-1, keepdim=True)
        goal = torch.randn(2, 3)
        with torch.no_grad():
            expected = actor(image, x, goal).numpy()
        actual = session.run(
            None, dict(perception=image.numpy(), state=x.numpy(), goal=goal.numpy())
        )[0]
        errors.append(float(np.max(np.abs(expected - actual))))
    if max(errors) > 1e-4:
        raise RuntimeError("ONNX parity failed")
    path.with_suffix(".export.json").write_text(
        json.dumps(
            dict(
                max_absolute_error=max(errors),
                cases=10,
                opset=17,
                inputs="ENU world state, FLU wxyz attitude, sensor-specific 2-channel image",
                output="four cubic acceleration/yaw-rate knots; duration .96 seconds",
                flight_validated=False,
            ),
            indent=2,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
