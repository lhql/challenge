#!/usr/bin/python3
"""Independent watchdog; only the control node owns the PX4 setpoint publisher."""

import json, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from mavros_msgs.msg import State, PositionTarget
from std_msgs.msg import String
from gap_nav.deployment.common import stamp, vector


class Safety(Node):
    def __init__(self):
        super().__init__("gap_safety")
        self.imu = None
        self.odom = None
        self.health = None
        self.state = None
        self.control = None
        self.times = {}
        self.setpoints = []
        self.pub = self.create_publisher(String, "/gap/safety", 10)
        self.create_subscription(
            Imu, "/livox/imu", lambda m: self.receive("imu", m), qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            "/gap/odom",
            lambda m: self.receive("odom", m),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/gap/inference_health", lambda m: self.receive("health", m), 10
        )
        self.create_subscription(
            String, "/gap/control_status", lambda m: self.receive("control", m), 10
        )
        self.create_subscription(
            State,
            "/mavros/state",
            lambda m: self.receive("state", m),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PositionTarget,
            "/mavros/mavros/local",
            self.on_setpoint,
            qos_profile_sensor_data,
        )
        self.create_timer(
            0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    def receive(self, key, msg):
        if key in ("health", "control"):
            try:
                msg = json.loads(msg.data)
            except (ValueError, TypeError):
                return
        setattr(self, key, msg)
        self.times[key] = time.monotonic()

    def on_setpoint(self, msg):
        now = time.monotonic()
        self.setpoints = [t for t in self.setpoints if now - t < 1]
        self.setpoints.append(now)

    def tick(self):
        now = time.monotonic()
        reasons = []
        for key, age in [("odom", 0.5), ("imu", 0.5), ("health", 0.6), ("state", 3.0)]:
            if now - self.times.get(key, -1e9) > age:
                reasons.append(key + " timeout")
        if self.health and not self.health.get("ready"):
            reasons.append("sensor/inference invalid")
        if self.odom:
            q = self.odom.pose.pose.orientation
            tilt = 1 - 2 * (q.x * q.x + q.y * q.y)
            if not np.isfinite(
                np.r_[
                    vector(self.odom.pose.pose.position),
                    vector(self.odom.twist.twist.linear),
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                ]
            ).all():
                reasons.append("nonfinite odometry")
            if tilt < np.cos(0.75):
                reasons.append("tilt limit")
            if np.linalg.norm(vector(self.odom.twist.twist.linear)) > 6.5:
                reasons.append("speed limit")
        if self.imu and (
            not np.isfinite(vector(self.imu.angular_velocity)).all()
            or np.linalg.norm(vector(self.imu.angular_velocity)) > 3.0
        ):
            reasons.append("angular rate limit")
        phase = self.control.get("phase", "WAIT") if self.control else "WAIT"
        if phase in ("TAKEOFF", "TRACK", "HOVER"):
            if now - self.times.get("control", -1e9) > 0.5:
                reasons.append("controller timeout")
            if len([t for t in self.setpoints if now - t < 1]) < 10:
                reasons.append("setpoint rate below 10 Hz")
            if self.state and (not self.state.armed or self.state.mode != "OFFBOARD"):
                reasons.append("OFFBOARD lost")
        publishers = [
            p
            for p in self.get_publishers_info_by_topic("/mavros/mavros/local")
            if p.topic_type
            in (
                "mavros_msgs/msg/PositionTarget",
                "geometry_msgs/msg/PoseStamped",
                "trajectory_msgs/msg/MultiDOFJointTrajectory",
            )
        ]
        if len(publishers) > 1:
            reasons.append("conflicting control publishers")
        self.pub.publish(
            String(
                data=json.dumps(
                    dict(ok=not reasons, reasons=reasons, stamp=time.time())
                )
            )
        )


def main():
    rclpy.init()
    node = Safety()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
