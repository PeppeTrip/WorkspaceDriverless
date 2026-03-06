from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    config_node = os.path.join(
        get_package_share_directory('clustering'),
        'config',
        'config.yaml'
    )

    clustering_node = Node(
        package='clustering',
        executable='clustering_node',
        name='clustering_node',
        output='screen',
        parameters=[
            config_node,
            {
                'input_topic': '/debug/lidar_points',
                'frame_id': 'hesai_lidar'
            }
        ]
    )

    debug_node = Node(
        package='clustering',
        executable='debug_pipeline_node',
        name='debug_pipeline_node',
        output='screen',
        parameters=[
            {
                'debug_input_topic': '/debug/lidar_points',
                'clusters_topic': '/perception/newclusters',
                'frame_id': 'hesai_lidar',
                'publish_hz': 5.0,
                'expected_near_min': 1,
                'expected_far_min': 1,
                'status_timeout_s': 2.0
            }
        ]
    )

    return LaunchDescription([clustering_node, debug_node])
