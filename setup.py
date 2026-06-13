from setuptools import find_packages, setup

package_name = 'navigation_methods'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ares',
    maintainer_email='ares@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "grid_nav=navigation_methods.grid_method:main",
            "rrt_nav=navigation_methods.rrt_method:main",
            "vpf_path_gen=navigation_methods.path_generator:main",
            "vpf_odometry=navigation_methods.puzzlebot_odometry:main",
            "vpf_nav=navigation_methods.puzzlebot_potential_field_controller:main",
        ],
    },
)
