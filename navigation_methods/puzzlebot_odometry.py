#!/usr/bin/env python3
import rclpy
import time
import math
from rclpy import qos
from rclpy.node import Node
from turtlesim.msg import Pose
from std_msgs.msg import Float32


class PuzzlebotOdometry(Node):
    def __init__(self):
        super().__init__("puzzlebot_odometry")

        self.get_logger().info("Nodo de odometría iniciado")
        self.r = 0.0505      # Radio de rueda [m]
        self.l = 0.183       # Distancia entre ruedas [m]
        self.rate = 100      # Frecuencia de odometría [Hz]
        self.wr = 0.0
        self.wl = 0.0
        self.v = 0.0
        self.w = 0.0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.pub_odom = self.create_publisher(Pose, "/odom", 10)

        self.create_subscription(
            Float32,
            "/VelocityEncR",
            self.cb_wR,
            qos.qos_profile_sensor_data
        )

        self.create_subscription(
            Float32,
            "/VelocityEncL",
            self.cb_wL,
            qos.qos_profile_sensor_data
        )

        self.t0 = time.time()
        self.create_timer(1.0 / self.rate, self.cb_odometry)

    def cb_wR(self, msg):
        self.wr = msg.data

    def cb_wL(self, msg):
        self.wl = msg.data

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def cb_odometry(self):
        now = time.time()
        dt = now - self.t0
        self.t0 = now

        self.v = (self.r / 2.0) * (self.wr + self.wl)
        self.w = (self.r / self.l) * (self.wr - self.wl)

        if abs(self.w) < 0.0001:
            self.x += dt * self.v * math.cos(self.theta)
            self.y += dt * self.v * math.sin(self.theta)
        else:
            theta_new = self.theta + self.w * dt

            self.x += (self.v / self.w) * (
                math.sin(theta_new) - math.sin(self.theta)
            )

            self.y -= (self.v / self.w) * (
                math.cos(theta_new) - math.cos(self.theta)
            )

            self.theta = theta_new

        self.theta = self.normalize_angle(self.theta)

        msg = Pose()
        msg.x = self.x
        msg.y = self.y
        msg.theta = self.theta

        self.pub_odom.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotOdometry()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Nodo detenido por el usuario")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
