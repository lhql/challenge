#!/usr/bin/python3
"""Record real ROS messages and depth-camera video, never synthesize flight data."""

import argparse, json, time
from pathlib import Path
import cv2, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from mavros_msgs.msg import State


class Observer(Node):
    def __init__(self, out, demo_goal=None):
        super().__init__("gap_validation_recorder")
        self.out = out
        self.f = (out / "events.jsonl").open("w")
        self.rows = []
        self.video = None
        self.frames = 0
        self.done = False
        self.demo_goal = demo_goal
        self.goal_sent = False
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        for topic in ["control_status", "safety", "inference_health"]:
            self.create_subscription(
                String, "/gap/" + topic, lambda m, t=topic: self.event(t, m), 10
            )
        self.create_subscription(
            Odometry, "/yopo_eval/body", self.truth, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "/camera/depth/image_rect_raw", self.image, qos_profile_sensor_data
        )
        self.create_subscription(
            State,
            "/mavros/state",
            lambda m: self.write(
                dict(type="fcu", armed=m.armed, mode=m.mode, connected=m.connected)
            ),
            qos_profile_sensor_data,
        )

    def write(self, row):
        row["wall"] = time.time()
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def event(self, kind, msg):
        try:
            row = json.loads(msg.data)
        except ValueError:
            return
        row["type"] = kind
        self.write(row)
        if (
            kind == "control_status"
            and row.get("phase") == "HOVER"
            and self.demo_goal is not None
            and not self.goal_sent
        ):
            self.goal_sent = True
            point = np.array(row["position"]) + np.array(self.demo_goal)
            msg = PoseStamped()
            msg.header.frame_id = "camera_init"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(
                float, point
            )
            msg.pose.orientation.w = 1.0
            self.goal_pub.publish(msg)
            self.write(
                dict(
                    type="test_goal_published",
                    position=point.tolist(),
                    topic="/goal_pose",
                )
            )
        if kind == "control_status" and row.get("phase") == "DONE":
            self.done = True

    def truth(self, msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation
        row = dict(
            type="truth",
            stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            p=[p.x, p.y, p.z],
            v=[v.x, v.y, v.z],
            q=[q.w, q.x, q.y, q.z],
        )
        self.rows.append(row)
        self.write(row)

    def image(self, msg):
        if msg.encoding != "32FC1":
            return
        dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
        a = np.ndarray(
            (msg.height, msg.width),
            dtype=dtype,
            buffer=bytes(msg.data),
            strides=(msg.step, 4),
        ).copy()
        a = np.clip(np.nan_to_num(a, nan=0, posinf=20), 0, 20)
        frame = cv2.applyColorMap((a / 20 * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        if self.video is None:
            self.video = cv2.VideoWriter(
                str(self.out / "actual_depth_stream.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20,
                (msg.width, msg.height),
            )
        self.video.write(frame)
        self.frames += 1

    def finish(self):
        if self.video:
            self.video.release()
        self.f.close()
        summary = dict(
            truth_samples=len(self.rows),
            video_frames=self.frames,
            max_altitude=max((r["p"][2] for r in self.rows), default=None),
            max_speed=max(
                (float(np.linalg.norm(r["v"])) for r in self.rows), default=None
            ),
            flight_passed=False,
            note="Raw observed metrics only; flight success requires event/trajectory review",
        )
        (self.out / "observed.json").write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--demo-goal", nargs=3, type=float)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Observer(args.output, args.demo_goal)
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
