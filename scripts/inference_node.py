#!/usr/bin/python3
"""Actor-only ONNX inference. Does not import Torch, MPC, Critic or shadows."""

import json, time
from collections import deque
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from livox_ros_driver2.msg import CustomMsg
from scipy.spatial.transform import Rotation
from gap_nav.perception.depth import preprocess
from gap_nav.perception.mid360 import project, deskew
from gap_nav.deployment.common import stamp, vector, evaluate


class Inference(Node):
    def __init__(self):
        super().__init__("gap_actor")
        for k, v in dict(
            sensor="depth",
            weights="",
            odom_topic="/gap/odom",
            depth_topic="/camera/depth/image_rect_raw",
            lidar_topic="/livox/lidar",
        ).items():
            self.declare_parameter(k, v)
        gp = lambda k: self.get_parameter(k).value
        self.sensor = gp("sensor")
        if self.sensor not in ("depth", "mid360"):
            raise ValueError("Unknown sensor")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            gp("weights"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        expected = [2, 96, 160] if self.sensor == "depth" else [2, 32, 128]
        if self.session.get_inputs()[0].shape[1:] != expected:
            raise ValueError("Checkpoint sensor/shape mismatch")
        self.odom = None
        self.goal = None
        self.image = None
        self.clouds = deque(maxlen=4)
        self.image_stamp = -1e9
        self.omega = np.zeros(3)
        self.imu_stamp = -1e9
        self.poses = deque(maxlen=120)
        self.last = -1e9
        self.pub = self.create_publisher(String, "/gap/trajectory", 1)
        self.path_pub = self.create_publisher(Path, "/gap/path", 1)
        self.health = self.create_publisher(String, "/gap/inference_health", 10)
        self.create_subscription(
            Odometry, gp("odom_topic"), self.on_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            Imu, "/livox/imu", self.on_imu, qos_profile_sensor_data
        )
        self.create_subscription(PoseStamped, "/gap/goal", self.on_goal, 10)
        if self.sensor == "depth":
            self.create_subscription(
                Image, gp("depth_topic"), self.on_depth, qos_profile_sensor_data
            )
        else:
            self.create_subscription(
                CustomMsg, gp("lidar_topic"), self.on_cloud, qos_profile_sensor_data
            )
        self.create_timer(0.05, self.tick)

    def on_goal(self, msg):
        if self.odom and msg.header.frame_id == self.odom.header.frame_id:
            self.goal = msg

    def on_imu(self, msg):
        self.omega = vector(msg.angular_velocity)
        self.imu_stamp = stamp(msg)

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        pos = vector(msg.pose.pose.position)
        quat = np.array([q.x, q.y, q.z, q.w])
        t = stamp(msg)
        if (
            not np.isfinite(np.r_[pos, quat]).all()
            or abs(np.linalg.norm(quat) - 1) > 0.05
        ):
            return
        if self.poses and t <= self.poses[-1][0]:
            return
        self.odom = msg
        self.poses.append((t, pos, quat))

    def on_depth(self, msg):
        if msg.encoding not in ("32FC1", "16UC1"):
            return
        dtype = (
            np.dtype(">f4" if msg.is_bigendian else "<f4")
            if msg.encoding == "32FC1"
            else np.dtype(">u2" if msg.is_bigendian else "<u2")
        )
        a = np.ndarray(
            (msg.height, msg.width),
            dtype=dtype,
            buffer=bytes(msg.data),
            strides=(msg.step, dtype.itemsize),
        ).astype(np.float32)
        if msg.encoding == "16UC1":
            a *= 0.001
        self.image = preprocess(a)
        self.image_stamp = stamp(msg)

    def on_cloud(self, msg):
        if msg.points:
            self.clouds.append(
                (msg, stamp(msg) + max(p.offset_time for p in msg.points) * 1e-9)
            )

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        start = time.perf_counter()
        error = "waiting for inputs"
        covered = [
            (msg, t) for msg, t in self.clouds if self.poses and t <= self.poses[-1][0]
        ]
        if self.sensor == "mid360" and covered and len(self.poses) > 1:
            msg, capture = covered[-1]
            points = np.array([[p.x, p.y, p.z] for p in msg.points])
            times = np.array([p.offset_time * 1e-9 + stamp(msg) for p in msg.points])
            keep = np.isfinite(points).all(1) & (np.linalg.norm(points, axis=1) > 0.1)
            if keep.any():
                try:
                    t = times[keep].max()
                    pt, p, q = zip(*self.poses)
                    self.image = project(deskew(points[keep], times[keep], pt, p, q, t))
                    self.image_stamp = t
                    self.clouds = deque(
                        [(m, ct) for m, ct in self.clouds if ct > capture], maxlen=4
                    )
                except ValueError:
                    error = "scan waiting for pose coverage"
        ready = (
            self.odom is not None
            and self.image is not None
            and -0.05 <= now - self.image_stamp < 0.35
            and -0.05 <= now - stamp(self.odom) < 0.25
            and -0.05 <= now - self.imu_stamp < 0.25
        )
        if ready and self.goal is not None:
            msg = self.odom
            q = msg.pose.pose.orientation
            quat = [q.x, q.y, q.z, q.w]
            vel = Rotation.from_quat(quat).apply(vector(msg.twist.twist.linear))
            x = np.r_[
                vector(msg.pose.pose.position), vel, q.w, q.x, q.y, q.z, self.omega
            ].astype(np.float32)
            goal = vector(self.goal.pose.position).astype(np.float32)
            knots = self.session.run(
                None,
                dict(
                    perception=self.image[None].astype(np.float32),
                    state=x[None],
                    goal=goal[None],
                ),
            )[0][0]
            ready = bool(np.isfinite(knots).all())
            if ready:
                path = Path()
                path.header = msg.header
                for t in np.linspace(0, 0.96, 17):
                    pos, _, _ = evaluate(x, knots, t)
                    pose = PoseStamped()
                    pose.header = path.header
                    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
                        map(float, pos)
                    )
                    pose.pose.orientation.w = 1.0
                    path.poses.append(pose)
                self.path_pub.publish(path)
                self.pub.publish(
                    String(
                        data=json.dumps(
                            dict(
                                stamp=now,
                                frame=msg.header.frame_id,
                                goal_stamp=stamp(self.goal),
                                x=x.tolist(),
                                knots=knots.tolist(),
                                duration=0.96,
                            )
                        )
                    )
                )
        elapsed = time.perf_counter() - start
        self.health.publish(
            String(
                data=json.dumps(
                    dict(
                        stamp=now,
                        ready=bool(ready and elapsed < 0.1),
                        sensor=self.sensor,
                        sensor_stamp=self.image_stamp,
                        odom_age=now - stamp(self.odom) if self.odom else None,
                        imu_age=now - self.imu_stamp,
                        inference_ms=elapsed * 1000,
                        reason="" if ready else error,
                    )
                )
            )
        )


def main():
    rclpy.init()
    node = Inference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
