#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation


class OpponentEKFTracker(Node):
    def __init__(self):
        super().__init__('opp_ekf_tracker')

        self.declare_parameter('measurement_topic', '/ego_racecar/opp_odom')
        self.declare_parameter('output_topic', '/ego_racecar/opp_odom_ekf')
        self.declare_parameter('output_pose_topic', '/ego_racecar/opp_odom_ekf_pose')
        self.declare_parameter('process_noise_pos', 0.05)
        self.declare_parameter('process_noise_vel', 0.5)
        self.declare_parameter('measurement_noise_pos', 0.08)
        self.declare_parameter('measurement_noise_vel', 0.8)

        measurement_topic = self.get_parameter('measurement_topic').value
        output_topic = self.get_parameter('output_topic').value
        output_pose_topic = self.get_parameter('output_pose_topic').value

        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, output_pose_topic, 10)
        self.sub = self.create_subscription(Odometry, measurement_topic, self.odom_callback, 10)

        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 0.5
        self.last_stamp = None
        self.initialized = False

        q_pos = float(self.get_parameter('process_noise_pos').value)
        q_vel = float(self.get_parameter('process_noise_vel').value)
        r_pos = float(self.get_parameter('measurement_noise_pos').value)
        r_vel = float(self.get_parameter('measurement_noise_vel').value)
        self.Q_base = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.diag([r_pos, r_pos, r_vel, r_vel])
        self.H = np.eye(4)

    def odom_callback(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        z = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.twist.twist.linear.x], [msg.twist.twist.linear.y]])

        if not self.initialized:
            self.x = z.copy()
            self.last_stamp = stamp
            self.initialized = True
            self.publish_estimate(msg.header.frame_id, msg.child_frame_id or 'opp_racecar/base_link', msg.header.stamp)
            return

        dt = max(stamp - self.last_stamp, 1e-3)
        self.last_stamp = stamp

        F = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        Q = self.Q_base * max(dt, 0.05)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.publish_estimate(msg.header.frame_id, msg.child_frame_id or 'opp_racecar/base_link', msg.header.stamp)

    def publish_estimate(self, frame_id, child_frame_id, stamp) -> None:
        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        yaw = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-4 else 0.0
        quat = Rotation.from_euler('z', yaw).as_quat()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = frame_id
        odom.child_frame_id = child_frame_id
        odom.pose.pose.position.x = float(self.x[0, 0])
        odom.pose.pose.position.y = float(self.x[1, 0])
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentEKFTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
