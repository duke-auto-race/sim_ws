import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share_directory = os.path.join(get_package_share_directory('mpc'), 'waypoints', '')
    waypoint_path = share_directory + 'Melbourne_map_mpc.csv'

    return LaunchDescription([
        Node(
            package='mpc',
            executable='mpc_node.py',
            name='opp_mpc_node',
            parameters=[{
                'waypoints_path': waypoint_path,
                'pose_topic': '/opp_racecar/odom',
                'drive_topic': '/opp_drive',
            }],
            output='screen',
        ),
        Node(
            package='mpc',
            executable='opponent_ekf_tracker.py',
            name='opp_ekf_tracker',
            parameters=[{
                'measurement_topic': '/ego_racecar/opp_odom',
                'output_topic': '/ego_racecar/opp_odom_ekf',
                'output_pose_topic': '/ego_racecar/opp_odom_ekf_pose',
            }],
            output='screen',
        ),
        Node(
            package='mpc',
            executable='ego_ekf_follower.py',
            name='ego_ekf_follower',
            parameters=[{
                'ego_odom_topic': '/ego_racecar/odom',
                'target_odom_topic': '/ego_racecar/opp_odom_ekf',
                'drive_topic': '/drive',
                'follow_distance': 1.0,
                'max_speed': 2.4,
                'min_speed': 0.8,
                'kp_dist': 1.6,
                'kp_yaw': 1.6,
            }],
            output='screen',
        ),
    ])
