# Navigation Methods

Repositorio con diferentes métodos de navegación implementados en ROS 2 y Python.  
El paquete incluye algoritmos de planeación de rutas y control de movimiento para robots móviles, principalmente usando `turtlesim` como entorno de simulación y nodos compatibles con una arquitectura tipo Puzzlebot.

Los métodos principales incluidos son:

- Navegación basada en grid con Dijkstra cinemático.
- Navegación basada en RRT.
- Navegación reactiva con campos potenciales virtuales.
- Generación de puntos objetivo.
- Odometría para el robot.
- Controlador de seguimiento de puntos.

---

## Contenido del repositorio

La estructura principal del repositorio es:

```bash
navigation_methods/
├── navigation_methods/
│   ├── __init__.py
│   ├── grid_method.py
│   ├── rrt_method.py
│   ├── path_generator.py
│   ├── puzzlebot_odometry.py
│   ├── puzzlebot_potential_field_controller.py
│   ├── mapatest.png
│   └── ruta_turtlesim.png
├── resource/
├── test/
├── package.xml
├── setup.py
├── setup.cfg
└── LICENSE
