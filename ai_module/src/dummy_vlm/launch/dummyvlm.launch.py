from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():

    ros1_bridge = Node(
        package='ros1_bridge',
        executable='dynamic_bridge',
        arguments=['--bridge-all-1to2-topics']
    )

    goalPub = Node(
        package='dummy_vlm',
        executable='sub.py',
    )

    return LaunchDescription([
        # world,
        # ros1_bridge,
        goalPub,
        # controller1,
        # controller2,
        # controller3,
    ])
