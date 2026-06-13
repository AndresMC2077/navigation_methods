#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import TeleportAbsolute, SetPen

import cv2
import numpy as np
import math
import random


TURTLESIM_MAX_X = 11.0
TURTLESIM_MAX_Y = 11.0

def generar_grid_desde_imagen(ruta_imagen, celdas_ancho=60, celdas_alto=60, radio_robot_px=12):
    img_gris = cv2.imread(ruta_imagen, 0)

    if img_gris is None:
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # Obstáculos
    _, mapa_binario = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radio_robot_px * 2, radio_robot_px * 2)
    )
    mapa_inflado = cv2.dilate(mapa_binario, kernel)

    grid = np.zeros((celdas_alto, celdas_ancho), dtype=np.uint8)

    bloque_x = img_gris.shape[1] // celdas_ancho
    bloque_y = img_gris.shape[0] // celdas_alto

    for j in range(celdas_alto):
        for i in range(celdas_ancho):
            sub_bloque = mapa_inflado[
                j * bloque_y: (j + 1) * bloque_y,
                i * bloque_x: (i + 1) * bloque_x
            ]

            if np.max(sub_bloque) == 255:
                grid[j, i] = 1

    return grid


def mundo_a_grid(x_mundo, y_mundo, celdas_ancho=60, celdas_alto=60):
    col = int((x_mundo / TURTLESIM_MAX_X) * celdas_ancho)
    fila = int(((TURTLESIM_MAX_Y - y_mundo) / TURTLESIM_MAX_Y) * celdas_alto)

    # Evitar índices fuera de rango por redondeos
    col = max(0, min(celdas_ancho - 1, col))
    fila = max(0, min(celdas_alto - 1, fila))

    return fila, col


def grid_a_mundo(fila, col, celdas_ancho=60, celdas_alto=60):
    x_mundo = (col + 0.5) * (TURTLESIM_MAX_X / celdas_ancho)
    y_mundo = TURTLESIM_MAX_Y - ((fila + 0.5) * (TURTLESIM_MAX_Y / celdas_alto))
    return x_mundo, y_mundo


def pixel_imagen_a_mundo(px_x, px_y, img_shape):
    x_ts = (px_x / img_shape[1]) * TURTLESIM_MAX_X
    y_ts = TURTLESIM_MAX_Y - ((px_y / img_shape[0]) * TURTLESIM_MAX_Y)
    return x_ts, y_ts


def mundo_a_pixel(x_mundo, y_mundo, ancho_px, alto_px):
    """
    Convierte coordenadas de Turtlesim a píxeles de imagen.
    Esto permite dibujar el árbol RRT sin deformarlo por la grid.
    """
    px = int((x_mundo / TURTLESIM_MAX_X) * ancho_px)
    py = int(((TURTLESIM_MAX_Y - y_mundo) / TURTLESIM_MAX_Y) * alto_px)

    px = max(0, min(ancho_px - 1, px))
    py = max(0, min(alto_px - 1, py))

    return px, py
    
#funciones de rrt
def punto_libre(grid, x, y, celdas_ancho, celdas_alto):
    if not (0.0 <= x <= TURTLESIM_MAX_X and 0.0 <= y <= TURTLESIM_MAX_Y):
        return False

    fila, col = mundo_a_grid(x, y, celdas_ancho, celdas_alto)

    return grid[fila, col] == 0


def segmento_libre(grid, p1, p2, celdas_ancho, celdas_alto, pasos=25):
    x1, y1 = p1
    x2, y2 = p2

    for i in range(pasos + 1):
        t = i / pasos

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        if not punto_libre(grid, x, y, celdas_ancho, celdas_alto):
            return False

    return True


def distancia(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def punto_mas_cercano(nodos, punto):
    mejor_indice = 0
    mejor_distancia = float("inf")

    for i, nodo in enumerate(nodos):
        d = distancia(nodo, punto)

        if d < mejor_distancia:
            mejor_distancia = d
            mejor_indice = i

    return mejor_indice


def avanzar_hacia(p_desde, p_hacia, paso):
    dx = p_hacia[0] - p_desde[0]
    dy = p_hacia[1] - p_desde[1]

    d = math.hypot(dx, dy)

    if d < paso:
        return p_hacia

    x_nuevo = p_desde[0] + paso * dx / d
    y_nuevo = p_desde[1] + paso * dy / d

    return x_nuevo, y_nuevo

#Planificador
def algoritmo_rrt(
    grid,
    inicio_mundo,
    meta_mundo,
    celdas_ancho,
    celdas_alto,
    max_iteraciones=10000,
    paso_rrt=0.35,
    probabilidad_meta=0.18,
    tolerancia_meta=0.35
):

    inicio = (inicio_mundo[0], inicio_mundo[1])
    meta = (meta_mundo[0], meta_mundo[1])

    if not punto_libre(grid, inicio[0], inicio[1], celdas_ancho, celdas_alto):
        print("El punto inicial está dentro de un obstáculo.")
        return None, []

    if not punto_libre(grid, meta[0], meta[1], celdas_ancho, celdas_alto):
        print("La meta está dentro de un obstáculo.")
        return None, []

    nodos = [inicio]
    padres = [-1]
    aristas = []

    for iteracion in range(max_iteraciones):

        # Con cierta probabilidad se intenta crecer hacia la meta
        if random.random() < probabilidad_meta:
            muestra = meta
        else:
            muestra = (
                random.uniform(0.1, TURTLESIM_MAX_X - 0.1),
                random.uniform(0.1, TURTLESIM_MAX_Y - 0.1)
            )

        indice_cercano = punto_mas_cercano(nodos, muestra)
        nodo_cercano = nodos[indice_cercano]

        nuevo_nodo = avanzar_hacia(nodo_cercano, muestra, paso_rrt)

        if not punto_libre(grid, nuevo_nodo[0], nuevo_nodo[1], celdas_ancho, celdas_alto):
            continue

        if not segmento_libre(grid, nodo_cercano, nuevo_nodo, celdas_ancho, celdas_alto):
            continue

        nodos.append(nuevo_nodo)
        padres.append(indice_cercano)
        aristas.append((nodo_cercano, nuevo_nodo))

        # Intentar conectar con la meta
        if distancia(nuevo_nodo, meta) < tolerancia_meta:
            if segmento_libre(grid, nuevo_nodo, meta, celdas_ancho, celdas_alto):

                nodos.append(meta)
                padres.append(len(nodos) - 2)
                aristas.append((nuevo_nodo, meta))

                ruta = []
                indice_actual = len(nodos) - 1

                while indice_actual != -1:
                    x, y = nodos[indice_actual]
                    ruta.append((x, y, 0.0))
                    indice_actual = padres[indice_actual]

                ruta.reverse()

                print(f"RRT encontró ruta en {iteracion + 1} iteraciones.")
                return ruta, aristas

    return None, aristas

#vizualización
def dibujar_grid_ocupacion(img_color, grid, celdas_ancho, celdas_alto):
    alto_px, ancho_px, _ = img_color.shape
    bx = ancho_px / celdas_ancho
    by = alto_px / celdas_alto
    overlay = img_color.copy()

    for fila in range(celdas_alto):
        for col in range(celdas_ancho):
            if grid[fila, col] == 1:
                cv2.rectangle(
                    overlay,
                    (int(col * bx), int(fila * by)),
                    (int((col + 1) * bx), int((fila + 1) * by)),
                    (180, 105, 255),
                    -1
                )

    cv2.addWeighted(overlay, 0.4, img_color, 0.6, 0, img_color)

    return img_color


def animar_planificacion_opencv(ruta_imagen, grid, ruta, aristas, celdas_ancho, celdas_alto):
    img_original = cv2.imread(ruta_imagen)

    if img_original is None:
        print("No se pudo cargar la imagen para visualización.")
        return

    alto_px, ancho_px, _ = img_original.shape

    # Imagen para mostrar solo el árbol
    img_arbol = img_original.copy()

    # Imagen para mostrar árbol + ruta final
    img_final = img_original.copy()

    img_arbol = dibujar_grid_ocupacion(
        img_arbol,
        grid,
        celdas_ancho,
        celdas_alto
    )

    img_final = dibujar_grid_ocupacion(
        img_final,
        grid,
        celdas_ancho,
        celdas_alto
    )

    cv2.namedWindow("RRT - Arbol de exploracion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RRT - Arbol de exploracion", 700, 700)

    # Dibujar árbol de exploración
    for idx, (p1, p2) in enumerate(aristas):
        px1, py1 = mundo_a_pixel(p1[0], p1[1], ancho_px, alto_px)
        px2, py2 = mundo_a_pixel(p2[0], p2[1], ancho_px, alto_px)

        cv2.line(
            img_arbol,
            (px1, py1),
            (px2, py2),
            (0, 180, 0),
            1
        )

        if idx % 50 == 0:
            cv2.imshow("RRT - Arbol de exploracion", img_arbol)
            cv2.waitKey(1)

    cv2.imshow("RRT - Arbol de exploracion", img_arbol)
    cv2.waitKey(300)

    cv2.namedWindow("RRT - Ruta desde el arbol", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RRT - Ruta desde el arbol", 700, 700)

    # Dibujar árbol tenue en la imagen final
    for p1, p2 in aristas:
        px1, py1 = mundo_a_pixel(p1[0], p1[1], ancho_px, alto_px)
        px2, py2 = mundo_a_pixel(p2[0], p2[1], ancho_px, alto_px)

        cv2.line(
            img_final,
            (px1, py1),
            (px2, py2),
            (170, 220, 170),
            1
        )

    if ruta:
        for i in range(len(ruta) - 1):
            x1, y1, _ = ruta[i]
            x2, y2, _ = ruta[i + 1]

            px1, py1 = mundo_a_pixel(x1, y1, ancho_px, alto_px)
            px2, py2 = mundo_a_pixel(x2, y2, ancho_px, alto_px)

            cv2.line(
                img_final,
                (px1, py1),
                (px2, py2),
                (0, 0, 255),
                3
            )

        # Inicio y meta
        inicio_px = mundo_a_pixel(ruta[0][0], ruta[0][1], ancho_px, alto_px)
        meta_px = mundo_a_pixel(ruta[-1][0], ruta[-1][1], ancho_px, alto_px)

        cv2.circle(img_final, inicio_px, 7, (0, 255, 255), -1)
        cv2.circle(img_final, meta_px, 7, (255, 0, 0), -1)

    cv2.imshow("RRT - Ruta desde el arbol", img_final)

    print("-> Planificación lista.")
    print("-> Ventana 1: árbol de exploración RRT.")
    print("-> Ventana 2: ruta reconstruida desde el árbol.")
    print("-> Haz clic en una ventana de OpenCV y presiona cualquier tecla para mover el robot...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

class TurtlesimPlannerNode(Node):
    def __init__(self, ruta_calculada, grid, start_pose, ruta_imagen, radio_robot_px):
        super().__init__("turtlesim_rrt_planner_node")

        self.ruta = ruta_calculada
        self.grid = grid
        self.current_pose = None
        self.indice_ruta = 0
        self.vel_pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.pose_sub = self.create_subscription(
            Pose,
            "/turtle1/pose",
            self.pose_callback,
            10
        )

        self.teleport_cli = self.create_client(
            TeleportAbsolute,
            "/turtle1/teleport_absolute"
        )

        self.pen_cli = self.create_client(
            SetPen,
            "/turtle1/set_pen"
        )

        self.setup_simulacion(start_pose, ruta_imagen, radio_robot_px)

    def pose_callback(self, msg):
        self.current_pose = msg

    def call_service_sync(self, client, request):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Esperando servicio {client.srv_name}...")

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def setup_simulacion(self, start_pose, ruta_imagen, radio_robot_px):
        self.get_logger().info("Analizando muros del laberinto para dibujo vectorial...")

        img = cv2.imread(ruta_imagen)

        if img is None:
            raise FileNotFoundError(f"No se pudo abrir la imagen: {ruta_imagen}")

        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, mapa_binario = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radio_robot_px * 2, radio_robot_px * 2)
        )

        mapa_inflado = cv2.dilate(mapa_binario, kernel)

        contornos, _ = cv2.findContours(
            mapa_inflado,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        self.get_logger().info(f"Se detectaron {len(contornos)} estructuras de muros.")

        self.call_service_sync(self.pen_cli, SetPen.Request(off=1))
        muro_pen = SetPen.Request(r=255, g=0, b=127, width=3, off=0)

        for contorno in contornos:
            epsilon = 0.005 * cv2.arcLength(contorno, True)
            contorno_simplificado = cv2.approxPolyDP(contorno, epsilon, True)

            if len(contorno_simplificado) < 2:
                continue

            start_px = contorno_simplificado[0][0]
            start_ts_x, start_ts_y = pixel_imagen_a_mundo(
                start_px[0],
                start_px[1],
                img.shape
            )

            self.call_service_sync(
                self.teleport_cli,
                TeleportAbsolute.Request(
                    x=start_ts_x,
                    y=start_ts_y,
                    theta=0.0
                )
            )

            self.call_service_sync(self.pen_cli, muro_pen)

            for punto_wrap in contorno_simplificado[1:]:
                punto = punto_wrap[0]

                tx, ty = pixel_imagen_a_mundo(
                    punto[0],
                    punto[1],
                    img.shape
                )

                self.call_service_sync(
                    self.teleport_cli,
                    TeleportAbsolute.Request(
                        x=tx,
                        y=ty,
                        theta=0.0
                    )
                )

            self.call_service_sync(
                self.teleport_cli,
                TeleportAbsolute.Request(
                    x=start_ts_x,
                    y=start_ts_y,
                    theta=0.0
                )
            )

            self.call_service_sync(self.pen_cli, SetPen.Request(off=1))

        self.get_logger().info("Laberinto dibujado. Moviendo tortuga al inicio...")

        self.call_service_sync(
            self.teleport_cli,
            TeleportAbsolute.Request(
                x=start_pose[0],
                y=start_pose[1],
                theta=start_pose[2]
            )
        )

        self.call_service_sync(
            self.pen_cli,
            SetPen.Request(r=0, g=255, b=255, width=3, off=0)
        )

        self.get_logger().info("Iniciando navegación con ruta RRT...")
        self.timer = self.create_timer(0.05, self.ejecutar_controlador)

    def ejecutar_controlador(self):
        if self.current_pose is None:
            return

        if self.indice_ruta < len(self.ruta):
            target_x, target_y, _ = self.ruta[self.indice_ruta]

            dx = target_x - self.current_pose.x
            dy = target_y - self.current_pose.y

            distancia_error = math.hypot(dx, dy)

            if distancia_error < 0.18:
                self.indice_ruta += 1
                return

            angulo_meta = math.atan2(dy, dx)
            error_angulo = angulo_meta - self.current_pose.theta

            # Normalizar ángulo
            error_angulo = math.atan2(
                math.sin(error_angulo),
                math.cos(error_angulo)
            )

            msg = Twist()

            if abs(error_angulo) > 0.5:
                msg.linear.x = 0.2
            else:
                msg.linear.x = min(1.8 * distancia_error, 1.2)

            msg.angular.z = 5.0 * error_angulo
            self.vel_pub.publish(msg)

        else:
            self.vel_pub.publish(Twist())
            self.get_logger().info("¡Meta alcanzada con éxito usando RRT!")
            self.timer.cancel()
            raise SystemExit

def main(args=None):
    nombre_imagen = "/home/coote/ros2_ws/src/RRT/RRT/obstaculos.png"
    inicio_turtlesim = (6.0, 10.0, -math.pi / 2)
    meta_turtlesim = (1.0, 1.0)
    c_w = 60
    c_h = 60
    r_robot_px = 14

    try:
        print("1. Generando grid e inflando obstáculos...")

        mi_grid = generar_grid_desde_imagen(
            nombre_imagen,
            c_w,
            c_h,
            r_robot_px
        )

        print("2. Calculando ruta con RRT...")

        ruta, aristas = algoritmo_rrt(
            mi_grid,
            inicio_turtlesim,
            meta_turtlesim,
            c_w,
            c_h,
            max_iteraciones=10000,
            paso_rrt=0.35,
            probabilidad_meta=0.18,
            tolerancia_meta=0.35
        )

        if not ruta:
            print("No se pudo encontrar una ruta con RRT.")
            print("Prueba aumentar max_iteraciones, reducir paso_rrt o revisar que inicio/meta estén libres.")
            return

        print(f"Ruta generada desde el árbol RRT con {len(ruta)} puntos.")
        print("3. Mostrando planificación en OpenCV...")

        animar_planificacion_opencv(
            nombre_imagen,
            mi_grid,
            ruta,
            aristas,
            c_w,
            c_h
        )

        print("4. Iniciando nodo ROS 2...")

        rclpy.init(args=args)

        nodo = TurtlesimPlannerNode(
            ruta,
            mi_grid,
            inicio_turtlesim,
            nombre_imagen,
            r_robot_px
        )

        try:
            rclpy.spin(nodo)
        except SystemExit:
            pass

        nodo.destroy_node()
        rclpy.shutdown()

    except Exception as e:
        print(f"Error crítico en la simulación: {e}")


if __name__ == "__main__":
    main()
