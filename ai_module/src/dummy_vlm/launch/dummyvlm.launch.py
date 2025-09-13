from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import IncludeLaunchDescription ,DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution,LaunchConfiguration, PythonExpression
import os
from ament_index_python.packages import get_package_share_directory,get_package_prefix

def generate_launch_description():
    # share_dir = get_package_share_directory('hb_task2b')
    # pkg_sim_world = get_package_share_directory('hb_world')
    # pkg_sim_bot = get_package_share_directory('hb_bot')


     
    # world = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkg_sim_world, 'launch', 'world.launch.py'),
    #     )
    # )

    ros1_bridge = Node(
        package='ros1_bridge',
        executable='dynamic_bridge'
    )

    goalPub = Node(
            package='dummy_vlm',
            executable='sub.py',
        )
    # feedback = Node(
    #         package='hb_task2b',
    #         executable='feedback.py',
    #     )        
    
    
    return LaunchDescription([
        # world,
        # ros1_bridge,
        goalPub,
        # controller1,
        # controller2,
        # controller3,
        ])