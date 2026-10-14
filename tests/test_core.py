import unittest
import numpy as np
import torch
from gap_nav.dynamics import Physics, rollout, initial_memory, rotation
from gap_nav.trajectory import trajectory, fit_knots, references
from gap_nav.mpc.teacher import Teacher
from gap_nav.shadow.temporal import keyframes, inherit
from gap_nav.perception.mid360 import deskew, project
from gap_nav.learning.networks import Actor


class Core(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.x = torch.zeros(1, 13)
        self.x[:, 6] = 1
        self.boxes = torch.tensor([[[[4.0, -1.0, -1.0], [5.0, 1.0, 2.0]]]])

    def test_hover_and_attitude(self):
        X, U, M = rollout(self.x, torch.zeros(1, 12, 4))
        self.assertLess(X[..., :6].abs().max(), 1e-5)
        self.assertEqual(U.shape, (1, 12, 4, 4))
        self.assertEqual(X.shape, (1, 13, 13))
        self.assertTrue(torch.allclose(X[..., 6:10].norm(dim=-1), torch.ones(1, 13)))
        ref = torch.zeros(1, 12, 4)
        ref[..., 0] = 2
        X, _, _ = rollout(self.x, ref)
        self.assertGreater(X[0, -1, 0], 0.1)
        self.assertGreater(X[0, -1, 8].abs(), 0.01)

    def test_clone_keeps_actuator_and_delay(self):
        torch.manual_seed(5)
        ref = torch.randn(1, 12, 4)
        wind = torch.randn(1, 12, 3) * 0.05
        X, U, M = rollout(self.x, ref, wind)
        k = 5
        XX, UU, MM = rollout(X[:, k], ref[:, k:], wind[:, k:], (M[:, k, 0], M[:, k, 1]))
        self.assertTrue(torch.allclose(XX, X[:, k:], atol=1e-6))
        self.assertTrue(torch.allclose(UU, U[:, k:], atol=1e-6))
        wrong, _, _ = rollout(X[:, k], ref[:, k:], wind[:, k:])
        self.assertGreater((wrong - XX).abs().max(), 1e-4)

    def test_trajectory_continuity(self):
        self.x[:, 3] = 1.0
        knots = torch.ones(1, 4, 4)
        p, v, a, y = trajectory(self.x, knots)
        self.assertTrue(torch.allclose(p[:, 0], self.x[:, :3]))
        self.assertTrue(torch.allclose(v[:, 0], self.x[:, 3:6]))
        self.assertAlmostEqual(float(p[0, -1, 0]), 0.96 + 0.5 * 0.96**2, places=5)
        self.assertTrue(
            torch.allclose(fit_knots(references(self.x, knots)), knots, atol=1e-5)
        )

    def test_teacher_and_shadow(self):
        goal = torch.tensor([[3.0, 1.0, 0.0]])
        wind = torch.zeros(1, 12, 3)
        teacher = Teacher(iterations=2)
        base = teacher.solve(self.x, goal, self.boxes, wind=wind)
        self.assertEqual(base["X"].shape, (1, 13, 13))
        self.assertTrue(base["success"].all())
        keys = keyframes(base, self.boxes)
        self.assertTrue(((keys > 0) & (keys < 12)).all())
        improved, meta = inherit(teacher, base, goal, self.boxes, wind, int(keys[0, 0]))
        self.assertEqual(meta["clone_error"], 0)
        self.assertTrue(meta["future_only_first_unchanged"])
        self.assertTrue(torch.equal(improved["X"][:, 0], self.x))

    def test_warm_start_cannot_bypass_control_bounds(self):
        cfg = Physics(max_accel=1.5, max_speed=2.0)
        teacher = Teacher(cfg=cfg, iterations=0)
        result = teacher.solve(
            self.x,
            torch.tensor([[10.0, 0.0, 0.0]]),
            self.boxes,
            warm=torch.full((1, 12, 4), 9.0),
        )
        self.assertLessEqual(float(result["reference"][..., :3].abs().max()), 1.5)
        self.assertLessEqual(float(result["reference"][..., 3].abs().max()), 1.5)
        bad = self.x.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            teacher.solve(bad, torch.zeros(1, 3), self.boxes)

    def test_batched_inheritance_and_fixed_suffix(self):
        x = self.x.repeat(3, 1)
        x[:, 3] = torch.tensor([0.0, 0.4, 0.8])
        goal = torch.tensor([[3.0, 1.0, 0.0]]).repeat(3, 1)
        boxes = self.boxes.repeat(3, 1, 1, 1)
        wind = torch.randn(3, 12, 3) * 0.03
        teacher = Teacher(iterations=2)
        base = teacher.solve(x, goal, boxes, wind=wind)
        improved, info = inherit(teacher, base, goal, boxes, wind, 5)
        self.assertLess(info["clone_error"], 1e-5)
        self.assertEqual(len(info["returns_to_go"]), 3)
        self.assertTrue(torch.equal(improved["X"][:, 0], x))
        prefix = teacher.solve(
            x, goal, boxes, warm=base["reference"], wind=wind, freeze_after=5
        )
        self.assertTrue(
            torch.equal(prefix["reference"][:, 5:], base["reference"][:, 5:])
        )
        self.assertTrue((prefix["cost"] <= base["cost"] + 1e-6).all())

    def test_translation_equivariance(self):
        actor = Actor().eval()
        image = torch.rand(1, 2, 96, 160)
        goal = torch.tensor([[3.0, 1.0, 0.0]])
        shifted = self.x.clone()
        shifted[:, :3] += 50
        self.assertTrue(
            torch.allclose(
                actor(image, self.x, goal), actor(image, shifted, goal + 50), atol=1e-6
            )
        )

    def test_quaternion_sign_invariance(self):
        actor = Actor().eval()
        # Nonzero output layer makes the assertion meaningful.
        torch.nn.init.normal_(actor.head[-1].weight, std=0.1)
        image = torch.rand(1, 2, 96, 160)
        other = self.x.clone()
        other[:, 6:10] *= -1
        goal = torch.tensor([[2.0, 1.0, 1.0]])
        self.assertTrue(
            torch.allclose(
                actor(image, self.x, goal), actor(image, other, goal), atol=1e-6
            )
        )

    def test_velocity_tracker_preserves_coasting_and_brakes_at_envelope(self):
        from gap_nav.deployment.common import evaluate, bounded_velocity

        state = np.zeros(13)
        state[3] = 1.0
        p, v, acc = evaluate(state, np.zeros((4, 4)), 0.3)
        self.assertTrue(np.allclose(v, [1.0, 0.0, 0.0]))
        self.assertTrue(
            np.allclose(bounded_velocity(v, [0.0, 0.0, 1.6], [0.0, 0.0, 0.0]), v)
        )
        bounded = bounded_velocity([4.0, 3.0, 2.0], [0.0, 0.0, 2.2], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(bounded[:2]), 1.5)
        self.assertLessEqual(bounded[2], 0.1 + 1e-8)
        for z in [-2.0, 5.0]:
            v = bounded_velocity([0.0, 0.0, 2.0], [0.0, 0.0, z], [0.0, 0.0, 0.0])
            self.assertGreaterEqual(v[2], -0.3)
            self.assertLessEqual(v[2], 0.5)

    def test_motion_compensation_and_unknown(self):
        points = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        out = deskew(
            points,
            [0.0, 1.0],
            [0.0, 1.0],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 1]],
            1.0,
        )
        self.assertTrue(np.allclose(out[0], out[1]))
        image = project(out)
        self.assertEqual(image[1].sum(), 1)
        self.assertEqual(project(np.empty((0, 3))).sum(), 0)
        with self.assertRaises(ValueError):
            deskew(
                points,
                [-1.0, 0.0],
                [0.0, 1.0],
                [[0, 0, 0], [1, 0, 0]],
                [[0, 0, 0, 1], [0, 0, 0, 1]],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
