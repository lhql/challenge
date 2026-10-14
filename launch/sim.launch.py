"""Existing Astra full model + FAST-LIO/PX4 + single GAP controller."""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_prefix, get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory("astra_yopo")
    runner = os.path.join(
        get_package_prefix("astra_wall_control"), "lib/astra_wall_control/run_sim.sh"
    )
    common = {"use_sim_time": True}
    return LaunchDescription(
        [
            DeclareLaunchArgument("sensor", default_value="depth"),
            DeclareLaunchArgument("weights"),
            DeclareLaunchArgument("gui", default_value="false"),
            DeclareLaunchArgument("auto_arm", default_value="false"),
            DeclareLaunchArgument("auto_goal", default_value="true"),
            DeclareLaunchArgument("auto_land", default_value="true"),
            DeclareLaunchArgument("max_mission_s", default_value="60.0"),
            DeclareLaunchArgument("goal_x", default_value="4.0"),
            DeclareLaunchArgument("goal_y", default_value="0.0"),
            DeclareLaunchArgument("goal_z", default_value="0.0"),
            DeclareLaunchArgument("rootfs", default_value="/tmp/gap_nav_px4"),
            DeclareLaunchArgument(
                "world", default_value=os.path.join(share, "worlds/yopo400.world")
            ),
            SetEnvironmentVariable("ASTRA_WALL_ROOTFS", LaunchConfiguration("rootfs")),
            SetEnvironmentVariable("OPENBLAS_NUM_THREADS", "1"),
            ExecuteProcess(
                cmd=[
                    runner,
                    "my",
                    LaunchConfiguration("world"),
                    LaunchConfiguration("gui"),
                    share,
                    "4",
                ],
                output="screen",
            ),
            Node(
                package="mavros",
                executable="mavros_node",
                namespace="mavros",
                name="mavros",
                output="screen",
                parameters=[
                    dict(
                        common,
                        fcu_url="udp://:14540@localhost:14557",
                        gcs_url="",
                        target_system_id=1,
                        target_component_id=1,
                        fcu_protocol="v2.0",
                        plugin_denylist=[
                            "odometry",
                            "landing_target",
                            "safety_area",
                            "image_pub",
                            "vibration",
                            "distance_sensor",
                            "rangefinder",
                            "wheel_odometry",
                        ],
                    )
                ],
            ),
            TimerAction(
                period=15.0,
                actions=[
                    Node(
                        package="fast_lio",
                        executable="fastlio_mapping",
                        output="screen",
                        parameters=[
                            os.path.join(share, "config/mid360_sim.yaml"),
                            common,
                        ],
                    )
                ],
            ),
            Node(
                package="astra_yopo",
                executable="lio_bridge.py",
                output="screen",
                parameters=[common],
                remappings=[("/yopo/odom", "/gap/odom")],
            ),
            Node(
                package="shade_challenger",
                executable="inference_node.py",
                output="screen",
                parameters=[
                    common,
                    {
                        "sensor": LaunchConfiguration("sensor"),
                        "weights": LaunchConfiguration("weights"),
                    },
                ],
            ),
            Node(
                package="shade_challenger",
                executable="safety_node.py",
                output="screen",
                parameters=[common],
            ),
            Node(
                package="shade_challenger",
                executable="control_node.py",
                output="screen",
                parameters=[
                    common,
                    {
                        "auto_arm": ParameterValue(
                            LaunchConfiguration("auto_arm"), value_type=bool
                        ),
                        "sitl_rootfs": LaunchConfiguration("rootfs"),
                        "auto_goal": ParameterValue(
                            LaunchConfiguration("auto_goal"), value_type=bool
                        ),
                        "auto_land": ParameterValue(
                            LaunchConfiguration("auto_land"), value_type=bool
                        ),
                        "max_mission_s": ParameterValue(
                            LaunchConfiguration("max_mission_s"), value_type=float
                        ),
                        "goal_x": ParameterValue(
                            LaunchConfiguration("goal_x"), value_type=float
                        ),
                        "goal_y": ParameterValue(
                            LaunchConfiguration("goal_y"), value_type=float
                        ),
                        "goal_z": ParameterValue(
                            LaunchConfiguration("goal_z"), value_type=float
                        ),
                    },
                ],
            ),
        ]
    )
