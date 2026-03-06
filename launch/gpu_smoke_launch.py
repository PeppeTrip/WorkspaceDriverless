from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='clustering',
            executable='gpu_smoke_node',
            name='gpu_smoke_node',
            output='screen',
        )
    ])
