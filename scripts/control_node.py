#!/usr/bin/python3
"""Single MAVROS trajectory publisher with SITL-only automatic takeoff/landing."""

import json, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation
from gap_nav.deployment.common import (
    stamp,
    vector,
    evaluate,
    sitl_process,
    bounded_velocity,
)


class Control(Node):
    def __init__(self):
        super().__init__("gap_control")
        for k, v in dict(
            auto_arm=False,
            sitl_rootfs="/tmp/gap_nav_px4",
            takeoff_height=1.6,
            goal_x=4.0,
            goal_y=0.0,
            goal_z=0.0,
            max_mission_s=60.0,
            auto_goal=True,
            auto_land=True,
        ).items():
            self.declare_parameter(k, v)
        self.odom = None
        self.pose = None
        self.state = None
        self.safe = None
        self.plan = None
        self.goal = None
        self.goal_id = None
        self.goal_reached = False
        self.last_logged_phase = None
        self.times = {}
        self.phase = "WAIT"
        self.enter = time.monotonic()
        self.pending = None
        self.last_request = -1e9
        self.home = None
        self.hold = None
        self.align = np.eye(3)
        self.translation = np.zeros(3)
        self.yaw = 0.0
        self.settle = None
        self.reason = ""
        self.pub = self.create_publisher(PositionTarget, "/mavros/mavros/local", 1)
        self.status = self.create_publisher(String, "/gap/control_status", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/gap/goal", 10)
        self.arm = self.create_client(CommandBool, "/mavros/mavros/arming")
        self.mode = self.create_client(SetMode, "/mavros/set_mode")
        self.create_subscription(
            Odometry,
            "/gap/odom",
            lambda m: self.receive("odom", m),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            "/mavros/mavros/pose",
            lambda m: self.receive("pose", m),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            State,
            "/mavros/state",
            lambda m: self.receive("state", m),
            qos_profile_sensor_data,
        )
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.create_subscription(String, "/gap/safety", self.on_safe, 10)
        self.create_subscription(String, "/gap/trajectory", self.on_plan, 1)
        self.create_service(Trigger, "/gap/land", self.land)
        self.create_timer(
            0.033, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )

    def start_goal(self, point):
        self.goal = np.asarray(point, dtype=float)
        msg = PoseStamped()
        msg.header.frame_id = self.odom.header.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(
            float, self.goal
        )
        msg.pose.orientation.w = 1.0
        self.goal_id = stamp(msg)
        self.plan = None
        self.goal_reached = False
        self.hold = vector(self.odom.pose.pose.position)
        self.settle = None
        self.goal_pub.publish(msg)
        self.phase = "TRACK"
        self.enter = time.monotonic()
        self.get_logger().info("Camera Actor goal: " + str(self.goal.tolist()))

    def on_goal(self, msg):
        if (
            self.phase not in ("HOVER", "TRACK")
            or self.odom is None
            or msg.header.frame_id != self.odom.header.frame_id
        ):
            self.get_logger().warning("Wait for HOVER and use RViz frame camera_init")
            return
        point = vector(msg.pose.position)
        if point[2] == 0.0:
            point[2] = self.home[2] + float(self.get_parameter("takeoff_height").value)
        if (
            not np.isfinite(point).all()
            or np.linalg.norm(point[:2] - self.home[:2]) > 12
            or not self.home[2] + 0.7 <= point[2] <= self.home[2] + 2.3
        ):
            self.get_logger().warning("Goal outside the bounded demo workspace")
            return
        self.start_goal(point)

    def receive(self, key, m):
        setattr(self, key, m)
        self.times[key] = time.monotonic()

    def on_safe(self, msg):
        try:
            self.safe = json.loads(msg.data)
            self.times["safe"] = time.monotonic()
        except ValueError:
            pass

    def on_plan(self, msg):
        try:
            plan = json.loads(msg.data)
            k = np.asarray(plan["knots"])
            x = np.asarray(plan["x"])
            now = self.get_clock().now().nanoseconds * 1e-9
            if (
                k.shape == (4, 4)
                and x.shape == (13,)
                and np.isfinite(k).all()
                and np.isfinite(x).all()
                and plan["goal_stamp"] == self.goal_id
                and plan["frame"] == self.odom.header.frame_id
                and 0 <= now - plan["stamp"] < 0.25
                and plan["duration"] == 0.96
            ):
                self.plan = plan
        except (ValueError, KeyError, TypeError, AttributeError):
            pass

    def land(self, req, res):
        self.phase = "LAND"
        self.reason = "operator stop"
        res.success = True
        res.message = "Landing requested"
        return res

    def request(self, client, msg):
        t = time.monotonic()
        if self.pending:
            if self.pending.done() or t - self.last_request > 3:
                self.pending = None
        if (
            self.pending is None
            and t - self.last_request > 1
            and client.service_is_ready()
        ):
            self.pending = client.call_async(msg)
            self.last_request = t

    def publish(self, p, v, a, velocity_only=False):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        # MAVROS ROS input is ENU, conversion to NED is done by MAVROS exactly once.
        msg.type_mask = (
            PositionTarget.IGNORE_YAW_RATE
            | PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
        )
        if velocity_only:
            msg.type_mask |= (
                PositionTarget.IGNORE_PX
                | PositionTarget.IGNORE_PY
                | PositionTarget.IGNORE_PZ
            )
        p = self.translation + self.align @ p
        v = self.align @ v
        a = self.align @ a
        msg.position.x, msg.position.y, msg.position.z = map(float, p)
        msg.velocity.x, msg.velocity.y, msg.velocity.z = map(float, v)
        (
            msg.acceleration_or_force.x,
            msg.acceleration_or_force.y,
            msg.acceleration_or_force.z,
        ) = map(float, a)
        msg.yaw = float(self.yaw)
        self.pub.publish(msg)

    def tick(self):
        wall = time.monotonic()
        now = self.get_clock().now().nanoseconds * 1e-9
        ready = (
            all(
                wall - self.times.get(k, -1e9) < t
                for k, t in [("odom", 0.5), ("pose", 0.5), ("state", 3), ("safe", 0.3)]
            )
            and self.safe
            and self.safe.get("ok")
            and self.state.connected
        )
        if self.phase == "WAIT" and ready:
            self.home = vector(self.odom.pose.pose.position)
            self.hold = self.home.copy()
            self.phase = "PRESTREAM"
            self.enter = wall
        if self.phase == "PRESTREAM":
            if not ready:
                self.phase = "WAIT"
                return
            q = self.pose.pose.orientation
            lq = self.odom.pose.pose.orientation
            yaw_px = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
            yaw_lio = Rotation.from_quat([lq.x, lq.y, lq.z, lq.w]).as_euler("xyz")[2]
            self.align = Rotation.from_euler("z", yaw_px - yaw_lio).as_matrix()
            self.translation = vector(self.pose.pose.position) - self.align @ vector(
                self.odom.pose.pose.position
            )
            self.yaw = yaw_px
            self.publish(self.home, np.zeros(3), np.zeros(3))
            if wall - self.enter > 3 and self.get_parameter("auto_arm").value:
                if not self.get_parameter("use_sim_time").value or not sitl_process(
                    self.get_parameter("sitl_rootfs").value
                ):
                    self.reason = (
                        "Automatic arming requires local launch-owned SITL process"
                    )
                    return
                if self.state.mode != "OFFBOARD":
                    self.request(self.mode, SetMode.Request(custom_mode="OFFBOARD"))
                elif not self.state.armed:
                    self.request(self.arm, CommandBool.Request(value=True))
            if self.state.armed and self.state.mode == "OFFBOARD":
                self.phase = "TAKEOFF"
                self.enter = wall
                self.mission_start = wall
        elif self.phase in ("TAKEOFF", "TRACK", "HOVER"):
            if self.state is not None and (
                self.state.mode != "OFFBOARD" or not self.state.armed
            ):
                self.phase = "ABORT"
                self.reason = "Pilot/failsafe mode change; yielding control"
            elif not ready:
                self.phase = "LAND"
                self.reason = "Safety loss: " + str(self.safe)
            elif wall - self.mission_start > float(
                self.get_parameter("max_mission_s").value
            ):
                self.phase = "LAND"
                self.reason = "Mission timeout"
            else:
                p = vector(self.odom.pose.pose.position)
                v = vector(self.odom.twist.twist.linear)
                reference = self.hold.copy()
                velocity = np.zeros(3)
                acceleration = np.zeros(3)
                if self.phase == "TAKEOFF":
                    reference = self.home + np.array(
                        [0, 0, self.get_parameter("takeoff_height").value]
                    )
                    settled = (
                        np.linalg.norm(p - reference) < 0.15 and np.linalg.norm(v) < 0.2
                    )
                    self.settle = (
                        (wall if self.settle is None else self.settle)
                        if settled
                        else None
                    )
                    if self.settle and wall - self.settle > 2:
                        self.hold = reference.copy()
                        self.enter = wall
                        self.settle = None
                        if self.get_parameter("auto_goal").value:
                            self.start_goal(
                                reference
                                + np.array(
                                    [
                                        self.get_parameter("goal_" + axis).value
                                        for axis in "xyz"
                                    ]
                                )
                            )
                        else:
                            self.phase = "HOVER"
                            self.get_logger().info(
                                "Ready: click RViz 2D Goal Pose (camera_init)"
                            )
                elif self.phase == "TRACK":
                    self.reason = ""
                    direction = self.align @ (self.goal - p)
                    desired_yaw = np.arctan2(direction[1], direction[0])
                    yaw_error = np.arctan2(
                        np.sin(desired_yaw - self.yaw), np.cos(desired_yaw - self.yaw)
                    )
                    if np.linalg.norm(direction[:2]) > 1.0:
                        self.yaw += float(np.clip(yaw_error, -0.5 * 0.033, 0.5 * 0.033))
                    if self.plan and 0 <= now - self.plan["stamp"] < 0.3:
                        reference, velocity, acceleration = evaluate(
                            self.plan["x"],
                            self.plan["knots"],
                            now - self.plan["stamp"] + 0.3,
                        )
                        velocity = bounded_velocity(velocity, p, self.home)
                        self.hold = p.copy()
                    if np.linalg.norm(p - self.goal) < 0.3 and np.linalg.norm(v) < 0.25:
                        self.phase = "HOVER"
                        self.goal_reached = True
                        self.enter = wall
                        self.hold = self.goal.copy()
                elif (
                    self.goal_reached
                    and self.get_parameter("auto_land").value
                    and wall - self.enter > 3
                ):
                    self.phase = "LAND"
                    self.reason = "Goal reached"
                self.publish(
                    reference,
                    velocity,
                    acceleration,
                    velocity_only=self.phase == "TRACK",
                )
        if self.phase == "LAND":
            # Mode requests do not compete with the single reference publisher.
            if self.state and self.state.armed and self.state.mode == "OFFBOARD":
                self.request(self.mode, SetMode.Request(custom_mode="AUTO.LAND"))
            elif self.state and not self.state.armed:
                self.phase = "DONE"
            elif self.state and self.state.mode != "AUTO.LAND":
                self.phase = "ABORT"
                self.reason = "Yielding to flight mode change"
        if self.phase != self.last_logged_phase:
            self.get_logger().info("GAP phase=" + self.phase + " " + self.reason)
            self.last_logged_phase = self.phase
        self.status.publish(
            String(
                data=json.dumps(
                    dict(
                        phase=self.phase,
                        goal_reached=self.goal_reached,
                        goal=self.goal.tolist() if self.goal is not None else None,
                        reason=self.reason,
                        stamp=now,
                        position=(
                            vector(self.odom.pose.pose.position).tolist()
                            if self.odom
                            else None
                        ),
                    )
                )
            )
        )


def main():
    rclpy.init()
    node = Control()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
