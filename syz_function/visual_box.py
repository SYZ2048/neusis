import configparser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyhocon import ConfigFactory

def visualize_bbox_and_center(conf_file):
    # Read configuration file
    f = open(conf_file)
    conf_text = f.read()
    conf = ConfigFactory.parse_string(conf_text)

    object_bbox_min = conf.get_list('mesh.object_bbox_min')
    object_bbox_max = conf.get_list('mesh.object_bbox_max')
    x_max = conf.get_float('mesh.x_max')
    x_min = conf.get_float('mesh.x_min')
    y_max = conf.get_float('mesh.y_max')
    y_min = conf.get_float('mesh.y_min')
    z_max = conf.get_float('mesh.z_max')
    z_min = conf.get_float('mesh.z_min')

    # Calculate cube_center
    cube_center = np.array([
        (x_max + x_min) / 2,
        (y_max + y_min) / 2,
        (z_max + z_min) / 2
    ])
    print(cube_center)

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot box
    object_x_min, object_y_min, object_z_min = object_bbox_min
    object_x_max, object_y_max, object_z_max = object_bbox_max

    # Vertices of the bounding box
    vertices = np.array([
        [object_x_min, object_y_min, object_z_min],
        [object_x_max, object_y_min, object_z_min],
        [object_x_max, object_y_max, object_z_min],
        [object_x_min, object_y_max, object_z_min],
        [object_x_min, object_y_min, object_z_max],
        [object_x_max, object_y_min, object_z_max],
        [object_x_max, object_y_max, object_z_max],
        [object_x_min, object_y_max, object_z_max]
    ])

    # Edges of the bounding box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color='b')

    # Plot defined bounding box
    x_box = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ]


    # Plot edges of the defined bounding box
    for box in [x_box]:
        ax.plot3D(box[0], box[1], box[2], color='g', linestyle='--')

    # Plot cube center
    ax.scatter(cube_center[0], cube_center[1], cube_center[2], color='r', s=100, label='Cube Center')
    ax.scatter(object_bbox_min[0], object_bbox_min[1], object_bbox_min[2], color='g', s=100, label='object_bbox_min')
    ax.scatter(object_bbox_max[0], object_bbox_max[1], object_bbox_max[2], color='g', s=100, label='object_bbox_max')
    ax.scatter(x_max, y_max, z_max, color='m', s=100, label='MAX')
    ax.scatter(x_min, y_min, z_min, color='c', s=100, label='MIN')

    # Labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bounding Box and Cube Center Visualization')
    ax.legend()

    plt.show()

# Example usage
# conf_file = r'..\confs\14deg_planeFull.conf'  # Replace with your configuration file path
conf_file = r'..\confs\plane_vertical_raw.conf'
visualize_bbox_and_center(conf_file)