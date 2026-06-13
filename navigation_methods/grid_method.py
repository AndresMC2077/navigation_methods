#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import TeleportAbsolute, SetPen
import cv2
import numpy as np
import heapq
import math

# =========================================================================
# 1. PREPARACIÓN DEL ENTORNO Y CONVERSIONES
# =========================================================================
def generar_grid_desde_imagen(ruta_imagen, celdas_ancho=60, celdas_alto=60, radio_robot_px=12):
    img_gris = cv2.imread(ruta_imagen, 0) 
    if img_gris is None:
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")
    
    _, mapa_binario = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radio_robot_px * 2, radio_robot_px * 2))
    mapa_inflado = cv2.dilate(mapa_binario, kernel)

    grid = np.zeros((celdas_alto, celdas_ancho), dtype=np.uint8)
    bloque_x = img_gris.shape[1] // celdas_ancho
    bloque_y = img_gris.shape[0] // celdas_alto

    for j in range(celdas_alto):
        for i in range(celdas_ancho):
            sub_bloque = mapa_inflado[j * bloque_y : (j + 1) * bloque_y, 
                                      i * bloque_x : (i + 1) * bloque_x]
            if np.max(sub_bloque) == 255:
                grid[j, i] = 1
    return grid

TURTLESIM_MAX_X = 11.0
TURTLESIM_MAX_Y = 11.0

def mundo_a_grid(x_mundo, y_mundo, celdas_ancho=60, celdas_alto=60):
    col = int((x_mundo / TURTLESIM_MAX_X) * celdas_ancho)
    fila = int(((TURTLESIM_MAX_Y - y_mundo) / TURTLESIM_MAX_Y) * celdas_alto)
    return fila, col

def grid_a_mundo(fila, col, celdas_ancho=60, celdas_alto=60):
    x_mundo = (col + 0.5) * (TURTLESIM_MAX_X / celdas_ancho)
    y_mundo = TURTLESIM_MAX_Y - ((fila + 0.5) * (TURTLESIM_MAX_Y / celdas_alto))
    return x_mundo, y_mundo

def pixel_imagen_a_mundo(px_x, px_y, img_shape):
    # Convierte píxeles puros de la imagen a coordenadas Turtlesim
    x_ts = (px_x / img_shape[1]) * TURTLESIM_MAX_X
    y_ts = TURTLESIM_MAX_Y - ((px_y / img_shape[0]) * TURTLESIM_MAX_Y)
    return x_ts, y_ts

# =========================================================================
# 2. PLANIFICADOR DIJKSTRA CINEMÁTICO CON SUB-STEPPING (SOLUCIÓN ESQUINAS)
# =========================================================================
def algoritmo_dijkstra_cinematico(grid, inicio_mundo, meta_mundo, celdas_ancho, celdas_alto):
    # Controles un poco más suaves para Turtlesim
    CONTROLES = [(1.0, 0.0), (0.7, 1.2), (0.7, -1.2), (0.0, 2.0), (0.0, -2.0)]
    DT = 0.5 # Paso de tiempo aumentado ligeramente para eficiencia
    TOLERANCIA = 0.3

    open_list = []
    counter = 0
    heapq.heappush(open_list, (0.0, counter, inicio_mundo))
    
    past_cost = {inicio_mundo: 0.0}
    padres = {inicio_mundo: None} 
    # Espacio de configuración Discretizado (Fila, Col, Theta)
    visitados_cspace = np.zeros((celdas_alto, celdas_ancho, 8), dtype=bool)
    nodos_expandidos = []

    while open_list and counter < 150000:
        costo_actual, _, actual = heapq.heappop(open_list)
        x, y, theta = actual

        if math.hypot(x - meta_mundo[0], y - meta_mundo[1]) < TOLERANCIA:
            camino = []
            while actual is not None:
                camino.append(actual)
                actual = padres[actual]
            return camino[::-1], nodos_expandidos

        f, c = mundo_a_grid(x, y, celdas_ancho, celdas_alto)
        theta_disc = int(((theta % (2 * math.pi)) / (2 * math.pi)) * 8) % 8
        
        if not (0 <= f < celdas_alto and 0 <= c < celdas_ancho) or visitados_cspace[f, c, theta_disc]: 
            continue

        visitados_cspace[f, c, theta_disc] = True
        nodos_expandidos.append((f, c))
        counter += 1
        
        # Expandir los controles discretos
        for v, w in CONTROLES:
            # SOLUCIÓN CORTAR ESQUINAS: Submuestreo estricto
            pasos_verificacion = 4
            dt_sub = DT / pasos_verificacion
            colision = False
            
            x_check, y_check, theta_check = x, y, theta
            
            for _ in range(pasos_verificacion):
                if w == 0.0:
                    x_check += v * math.cos(theta_check) * dt_sub
                    y_check += v * math.sin(theta_check) * dt_sub
                else:
                    x_check += (v / w) * (math.sin(theta_check + w * dt_sub) - math.sin(theta_check))
                    y_check -= (v / w) * (math.cos(theta_check + w * dt_sub) - math.cos(theta_check))
                
                theta_check = (theta_check + w * dt_sub) % (2 * math.pi)
                
                # Validar colisión en este punto intermedio
                f_check, c_check = mundo_a_grid(x_check, y_check, celdas_ancho, celdas_alto)
                if not (0 <= f_check < celdas_alto and 0 <= c_check < celdas_ancho) or grid[f_check, c_check] != 0:
                    colision = True
                    break
            
            if colision: continue

            # Estado final de la acción (si es segura)
            nuevo_estado = (x_check, y_check, theta_check)
            f_new, c_new = mundo_a_grid(x_check, y_check, celdas_ancho, celdas_alto)

            # Penalizar giros excesivos
            costo_accion = DT + (abs(w) * 0.15)
            costo_total = costo_actual + costo_accion
            
            if nuevo_estado not in past_cost or costo_total < past_cost[nuevo_estado]:
                past_cost[nuevo_estado] = costo_total
                padres[nuevo_estado] = actual
                heapq.heappush(open_list, (costo_total, counter, nuevo_estado))        

    return None, []

# =========================================================================
# 3. VISUALIZACIÓN EN OPENCV
# =========================================================================
def animar_planificacion_opencv(ruta_imagen, grid, ruta, nodos_expandidos, celdas_ancho, celdas_alto):
    img_color = cv2.imread(ruta_imagen)
    alto_px, ancho_px, _ = img_color.shape
    bx, by = ancho_px / celdas_ancho, alto_px / celdas_alto

    overlay = img_color.copy()
    for j in range(celdas_alto):
        for i in range(celdas_ancho):
            if grid[j, i] == 1: # Muros inflados
                cv2.rectangle(overlay, (int(i*bx), int(j*by)), (int((i+1)*bx), int((j+1)*by)), (180, 105, 255), -1) # Rosa claro
    cv2.addWeighted(overlay, 0.4, img_color, 0.6, 0, img_color)

    for idx, (f, c) in enumerate(nodos_expandidos):
        cv2.rectangle(img_color, (int(c*bx), int(f*by)), (int((c+1)*bx), int((f+1)*by)), (100, 255, 100), -1) # Verde
        if idx % 150 == 0:
            cv2.imshow("Fase de Planificacion", img_color)
            cv2.waitKey(1)
    
    if ruta:
        for i in range(len(ruta) - 1):
            f1, c1 = mundo_a_grid(ruta[i][0], ruta[i][1], celdas_ancho, celdas_alto)
            f2, c2 = mundo_a_grid(ruta[i+1][0], ruta[i+1][1], celdas_ancho, celdas_alto)
            # Centro de la celda
            cv2.line(img_color, (int(c1*bx+bx/2), int(f1*by+by/2)), (int(c2*bx+bx/2), int(f2*by+by/2)), (255, 150, 0), 2) # Azul claro
            
        cv2.imshow("Fase de Planificacion", img_color)
        print("-> Planificación lista. Haz clic en la imagen y presiona cualquier tecla para mover el robot...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# =========================================================================
# 4. NODO ROS 2 (DIBUJO DE MUROS Y CONTROL)
# =========================================================================
class TurtlesimPlannerNode(Node):
    def __init__(self, ruta_calculada, grid, start_pose, ruta_imagen, radio_robot_px):
        super().__init__('turtlesim_planner_node')
        self.ruta = ruta_calculada
        
        # Publicador y Suscriptor
        self.vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.current_pose = None
        self.indice_ruta = 0
        
        # Servicios
        self.teleport_cli = self.create_client(TeleportAbsolute, '/turtle1/teleport_absolute')
        self.pen_cli = self.create_client(SetPen, '/turtle1/set_pen')

        # Ejecutar secuencia de setup
        self.setup_simulacion(start_pose, ruta_imagen, radio_robot_px)

    def pose_callback(self, msg):
        self.current_pose = msg

    def call_service_sync(self, client, request):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Esperando servicio {client.srv_name}...')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def setup_simulacion(self, start_pose, ruta_imagen, radio_robot_px):
        # 1. GENERACIÓN DE SILUETAS VECTORIALES DE LOS MUROS
        self.get_logger().info('Analizando muros del laberinto para dibujo vectorial...')
        img = cv2.imread(ruta_imagen)
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar muros originales y expandir (C-Space)
        _, mapa_binario = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radio_robot_px * 2, radio_robot_px * 2))
        mapa_inflado = cv2.dilate(mapa_binario, kernel)
        
        # Obtener los contornos exactos del C-Space
        # Usamos Aproximación Simple para reducir puntos redundantes
        contornos, _ = cv2.findContours(mapa_inflado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.get_logger().info(f'Se detectaron {len(contornos)} estructuras de muros. Iniciando dibujo en Turtlesim...')

        # 2. FASE "ARQUITECTO": Dibujar el laberinto
        # Levantar lápiz para teletransporte inicial
        self.call_service_sync(self.pen_cli, SetPen.Request(off=1))

        # Configuración del lápiz para muros (Rosa fucsia intenso)
        muro_pen = SetPen.Request(r=255, g=0, b=127, width=3, off=0)

        for contorno in contornos:
            # Simplificar aún más el contorno (Ramer-Douglas-Peucker) para reducir líneas
            epsilon = 0.005 * cv2.arcLength(contorno, True)
            contorno_simplificado = cv2.approxPolyDP(contorno, epsilon, True)
            
            if len(contorno_simplificado) < 2: continue
            
            # Teletransportarse al primer punto (con lápiz arriba)
            start_px = contorno_simplificado[0][0]
            start_ts_x, start_ts_y = pixel_imagen_a_mundo(start_px[0], start_px[1], img.shape)
            
            self.call_service_sync(self.teleport_cli, TeleportAbsolute.Request(x=start_ts_x, y=start_ts_y, theta=0.0))
            
            # Bajar lápiz y dibujar contorno completo
            self.call_service_sync(self.pen_cli, muro_pen)
            
            # Iterar por el resto de puntos (excluyendo el primero)
            for punto_wrap in contorno_simplificado[1:]:
                punto = punto_wrap[0]
                tx, ty = pixel_imagen_a_mundo(punto[0], punto[1], img.shape)
                self.call_service_sync(self.teleport_cli, TeleportAbsolute.Request(x=tx, y=ty, theta=0.0))
                
            # Cerrar el contorno volviendo al inicio
            self.call_service_sync(self.teleport_cli, TeleportAbsolute.Request(x=start_ts_x, y=start_ts_y, theta=0.0))
            
            # Levantar lápiz para saltar al siguiente muro
            self.call_service_sync(self.pen_cli, SetPen.Request(off=1))

        self.get_logger().info('Laberinto dibujado. Moviendo robot al punto de inicio...')

        # 3. FASE "PILOTO": Mover a la posición de inicio y preparar trayectoria
        # Teletransportar con lápiz arriba
        self.call_service_sync(self.teleport_cli, TeleportAbsolute.Request(x=start_pose[0], y=start_pose[1], theta=start_pose[2]))
        
        # Configurar lápiz para trayectoria (Azul cian claro)
        self.call_service_sync(self.pen_cli, SetPen.Request(r=0, g=255, b=255, width=3, off=0))

        # 4. Iniciar controlador de lazo cerrado (20Hz)
        self.get_logger().info('Iniciando navegación con Lazo Cerrado...')
        self.timer = self.create_timer(0.05, self.ejecutar_controlador)

    def ejecutar_controlador(self):
        if self.current_pose is None: return
            
        if self.indice_ruta < len(self.ruta):
            # Obtener el nodo objetivo actual de la ruta (Dijkstra)
            target_x, target_y, _ = self.ruta[self.indice_ruta]
            
            # Cálculo de errores con Pose actual (Sensores)
            dx = target_x - self.current_pose.x
            dy = target_y - self.current_pose.y
            distancia_error = math.hypot(dx, dy)

            # Si ya estamos cerca de este nodo, pasar al siguiente en la lista
            # Tolerancia un poco más estricta para clavar la ruta azul
            if distancia_error < 0.18: 
                self.indice_ruta += 1
                return

            # Cálculo de volante (Ángulo)
            angulo_meta = math.atan2(dy, dx)
            error_angulo = angulo_meta - self.current_pose.theta
            # Normalizar ángulo (-pi a pi)
            error_angulo = math.atan2(math.sin(error_angulo), math.cos(error_angulo))

            # Controlador Proporcional Local
            msg = Twist()
            # Velocidad lineal basada en la distancia, limitada a 1.2
            msg.linear.x = min(1.8 * distancia_error, 1.2) 
            # Velocidad angular basada en el error de volante
            msg.angular.z = 5.0 * error_angulo 
            
            self.vel_pub.publish(msg)
        else:
            # Freno final al llegar a la meta
            self.vel_pub.publish(Twist()) 
            self.get_logger().info('¡Meta alcanzada con éxito! Fin del ejercicio.')
            self.timer.cancel()
            raise SystemExit 

# ==========================================
# MAIN ROS 2
# ==========================================
def main(args=None):
    # Ruta absoluta de tu imagen (Modifícalo si cambiaste de carpeta)
    nombre_imagen = "/home/ares/ros2_ws/src/navigation_methods/navigation_methods/mapatest.png" 
    
    # Coordenadas reales Turtlesim (0 a 11)
    inicio_turtlesim = (6.0, 10.0, -math.pi/2) # Mirando hacia abajo
    meta_turtlesim = (1.0, 1.0)              
    
    # Dimensiones de la rejilla de ocupación (mantenemos 60x60 para eficiencia)
    c_w, c_h = 60, 60
    # Radio de seguridad del robot en píxeles
    r_robot_px = 14

    try:
        # FASE 1: Cálculo del Dijkstra Cinemático (Antes de ROS)
        print("1. Generando Grid y analizando C-Space...")
        mi_grid = generar_grid_desde_imagen(nombre_imagen, c_w, c_h, r_robot_px)
        
        print("2. Calculando trayectoria óptima cinemática...")
        ruta, exp = algoritmo_dijkstra_cinematico(mi_grid, inicio_turtlesim, meta_turtlesim, c_w, c_h)

        if not ruta:
            print("No se pudo encontrar una ruta viable respetando la cinemática de la tortuga.")
            return

        # FASE 2: Animación en OpenCV (Para rúbrica)
        animar_planificacion_opencv(nombre_imagen, mi_grid, ruta, exp, c_w, c_h)

        # FASE 3: Ejecución en ROS 2 Humble
        print(f"3. Ruta generada con {len(ruta)} puntos. Iniciando Nodo ROS 2 Humble...")
        rclpy.init(args=args)
        
        # Le pasamos la imagen original para que el nodo dibuje los muros
        nodo = TurtlesimPlannerNode(ruta, mi_grid, inicio_turtlesim, nombre_imagen, r_robot_px)
        
        try:
            rclpy.spin(nodo)
        except SystemExit:
            pass # Salida controlada al terminar la ruta

        nodo.destroy_node()
        rclpy.shutdown()
        
    except Exception as e:
        print(f"Error crítico en la simulación: {e}")

if __name__ == "__main__":
    main()
