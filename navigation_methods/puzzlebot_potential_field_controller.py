#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from std_msgs.msg import Bool


class PuzzlebotPotentialField(Node):
    def __init__(self):
        super().__init__("puzzlebot_potential_field_controller")
        self.get_logger().info("Esta vivo xdxdxd")
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_arrived = self.create_publisher(Bool, "/arrived", 10)
        self.create_subscription(Pose, "/odom", self.cb_odom, 10)
        self.create_subscription(Pose, "/next_point", self.cb_target, 10)
        self.create_timer(0.1, self.control_loop)

        # Pose actual del robot
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.got_odom = False

        # Objetivo actual
        self.goal_x = None
        self.goal_y = None
        self.got_target = False

        # Obstáculos
        self.obstacles = [
            [1.0, 1.0],
            [1.0, 2.0]
        ]

        # Parámetros del campo potencial
        self.k_att = 0.8
        self.k_rep = 0.45
        self.rep_radius = 0.3

        # Ganancias de movimiento
        self.Kv = 0.45
        self.Kw = 1.2

        # Límites físicos del Puzzlebot
        self.MAX_LIN_VEL = 0.16
        self.MAX_ANG_VEL = 1.1

        # Tolerancias
        self.goal_tolerance = 0.10
        self.angle_limit_to_move = 0.9

        # Estado de llegada
        self.arrived_state = False

    # Callbacks
    def cb_odom(self, msg):
        self.x = msg.x
        self.y = msg.y
        self.theta = msg.theta
        self.got_odom = True

    def cb_target(self, msg):
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.got_target = True
        self.arrived_state = False

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    # Saturación
    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    # -----------------------------
    # Detener robot
    # -----------------------------
    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.pub_cmd.publish(msg)

    # -----------------------------
    # Publicar llegada
    # -----------------------------
    def publish_arrived(self, value):
        msg = Bool()
        msg.data = value
        self.pub_arrived.publish(msg)

    # Fuerzas
    def attractive_force(self):
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        fx_att = self.k_att * dx
        fy_att = self.k_att * dy

        return fx_att, fy_att

    def repulsive_force(self):
        fx_rep_total = 0.0
        fy_rep_total = 0.0

        for obs in self.obstacles:
            obs_x = obs[0]
            obs_y = obs[1]

            dx = self.x - obs_x
            dy = self.y - obs_y

            distance = math.sqrt(dx**2 + dy**2)

            if distance < 0.001:
                distance = 0.001

            if distance <= self.rep_radius:
                ux = dx / distance
                uy = dy / distance

                rep_magnitude = self.k_rep * (
                    (1.0 / distance) - (1.0 / self.rep_radius)
                ) * (1.0 / (distance**2))

                fx_rep_total += rep_magnitude * ux
                fy_rep_total += rep_magnitude * uy

        return fx_rep_total, fy_rep_total

    def control_loop(self):
        if not self.got_odom or not self.got_target:
            return

        cmd = Twist()

        # Distancia al objetivo
        dx_goal = self.goal_x - self.x
        dy_goal = self.goal_y - self.y
        distance_to_goal = math.sqrt(dx_goal**2 + dy_goal**2)


        if distance_to_goal < self.goal_tolerance:
            self.stop_robot()

            if not self.arrived_state:
                self.get_logger().info(
                    f"Objetivo alcanzado: ({self.goal_x:.2f}, {self.goal_y:.2f})"
                )

                self.publish_arrived(True)
                self.arrived_state = True

            return
        else:
            self.publish_arrived(False)

        # Calculos de fuerzas
        fx_att, fy_att = self.attractive_force()
        fx_rep, fy_rep = self.repulsive_force()
        fx_total = fx_att + fx_rep
        fy_total = fy_att + fy_rep
        desired_theta = math.atan2(fy_total, fx_total)
        angle_error = self.normalize_angle(desired_theta - self.theta)
        force_magnitude = math.sqrt(fx_total**2 + fy_total**2)
        w = self.Kw * angle_error
        cmd.angular.z = self.clamp(
            w,
            -self.MAX_ANG_VEL,
            self.MAX_ANG_VEL
        )

        # Alineación del robot
        if abs(angle_error) < self.angle_limit_to_move:
            v = self.Kv * force_magnitude * math.cos(angle_error)

            cmd.linear.x = self.clamp(
                v,
                0.0,
                self.MAX_LIN_VEL
            )
        else:
            cmd.linear.x = 0.0

        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = PuzzlebotPotentialField()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Esta muerto, se cayo de un caballo")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()
if __name__ == "__main__":
    main()
