#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from std_msgs.msg import Bool


class PathGenerator(Node):
    def __init__(self):
        super().__init__("path_generator")

        self.get_logger().info("Generador de trayectoria iniciado")
        self.pub_point = self.create_publisher(Pose, "/next_point", 10)
        self.create_subscription(Bool, "/arrived", self.cb_arrived, 10)
        self.point_list = [
            [2.0, 2.0],
            [0.0, 2.0]
        ]

        self.current_index = 0
        self.waiting_for_arrival = False
        self.finished = False

        self.create_timer(0.2, self.loop)

    def cb_arrived(self, msg):
        if msg.data and self.waiting_for_arrival:
            self.get_logger().info("Punto alcanzado. Avanzando al siguiente.")
            self.current_index += 1
            self.waiting_for_arrival = False

    def loop(self):
        if self.finished:
            return

        if self.current_index >= len(self.point_list):
            self.get_logger().info("Trayectoria terminada")
            self.finished = True
            return

        x, y = self.point_list[self.current_index]

        msg = Pose()
        msg.x = x
        msg.y = y

        self.pub_point.publish(msg)
        self.waiting_for_arrival = True


def main(args=None):
    rclpy.init(args=args)
    node = PathGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Nodo detenido por el usuario")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
