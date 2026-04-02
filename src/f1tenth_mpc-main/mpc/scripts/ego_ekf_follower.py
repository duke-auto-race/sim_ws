#!/usr/bin/env python3
import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation


def wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class EgoEKFFollower(Node):
    def __init__(self):
        super().__init__('ego_ekf_follower')

        self.declare_parameter('ego_odom_topic', '/ego_racecar/odom')
        self.declare_parameter('target_odom_topic', '/ego_racecar/opp_odom_ekf')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('follow_distance', 1.5)
        self.declare_parameter('max_speed', 1.8)
        self.declare_parameter('min_speed', 0.3)
        self.declare_parameter('max_steer', 0.34)
        self.declare_parameter('kp_dist', 0.9)
        self.declare_parameter('kp_yaw', 1.8)
        self.declare_parameter('target_timeout', 0.3)

        ego_topic = self.get_parameter('ego_odom_topic').value
        target_topic = self.get_parameter('target_odom_topic').value
        drive_topic = self.get_parameter('drive_topic').value

        self.follow_distance = float(self.get_parameter('follow_distance').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.max_steer = float(self.get_parameter('max_steer').value)
        self.kp_dist = float(self.get_parameter('kp_dist').value)
        self.kp_yaw = float(self.get_parameter('kp_yaw').value)
        self.target_timeout = float(self.get_parameter('target_timeout').value)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_subscription(Odometry, ego_topic, self.ego_callback, 10)
        self.create_subscription(Odometry, target_topic, self.target_callback, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.ego = None
        self.target = None
        self.target_stamp = None

    def yaw_from_odom(self, odom: Odometry) -> float:
        q = odom.pose.pose.orientation
        quat = Rotation.from_quat([q.x, q.y, q.z, q.w])
        return quat.as_euler('zxy', degrees=False)[0]

    def ego_callback(self, msg: Odometry) -> None:
        self.ego = msg

    def target_callback(self, msg: Odometry) -> None:
        self.target = msg
        self.target_stamp = self.get_clock().now()

    def control_loop(self) -> None:
        drive = AckermannDriveStamped()

        if self.ego is None or self.target is None or self.target_stamp is None:
            drive.drive.speed = 0.0
            drive.drive.steering_angle = 0.0
            self.drive_pub.publish(drive)
            return

        age = (self.get_clock().now() - self.target_stamp).nanoseconds * 1e-9
        if age > self.target_timeout:
            drive.drive.speed = 0.0
            drive.drive.steering_angle = 0.0
            self.drive_pub.publish(drive)
            return

        ex = self.ego.pose.pose.position.x
        ey = self.ego.pose.pose.position.y
        tx = self.target.pose.pose.position.x
        ty = self.target.pose.pose.position.y
        ego_yaw = self.yaw_from_odom(self.ego)

        dx = tx - ex
        dy = ty - ey
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        yaw_error = wrap(target_heading - ego_yaw)

        speed_cmd = self.kp_dist * max(distance - self.follow_distance, 0.0)
        if distance > self.follow_distance + 0.1:
            speed_cmd = max(speed_cmd, self.min_speed)
        else:
            speed_cmd = 0.0
        speed_cmd = min(speed_cmd, self.max_speed)

        steer_cmd = self.kp_yaw * yaw_error
        steer_cmd = max(min(steer_cmd, self.max_steer), -self.max_steer)

        drive.drive.speed = float(speed_cmd)
        drive.drive.steering_angle = float(steer_cmd)
        self.drive_pub.publish(drive)


def main(args=None):
    rclpy.init(args=args)
    node = EgoEKFFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
