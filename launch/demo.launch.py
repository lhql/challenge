"""Forest demo: MID360 FAST-LIO localization, Depth GAP Actor, existing PX4 body."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    own = get_package_share_directory("shade_challenger")
    environment = get_package_share_directory("astra_yopo")
    defaults = dict(
        gui="true",
        rviz="true",
        auto_arm="true",
        auto_goal="false",
        auto_land="false",
        max_mission_s="300.0",
        goal_x="2.0",
        goal_y="0.0",
        goal_z="0.0",
        weights=os.path.join(own, "weights/depth.onnx"),
        world=os.path.join(environment, "worlds/jungle.world"),
    )
    arguments = {key: LaunchConfiguration(key) for key in defaults if key != "rviz"}
    arguments["sensor"] = "depth"
    return LaunchDescription(
        [
            *[
                DeclareLaunchArgument(key, default_value=value)
                for key, value in defaults.items()
            ],
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(own, "launch/sim.launch.py")
                ),
                launch_arguments=arguments.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                parameters=[{"use_sim_time": True}],
                arguments=["-d", os.path.join(own, "config/demo.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
