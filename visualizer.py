import tensorflow as tf
import os
import math
import numpy as np
import itertools
from multiprocessing import Pool

import open3d as o3d
import random
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

type_dict = {
    "TYPE_VEHICLE": 1,
    "TYPE_PEDESTRIAN": 2,
    "TYPE_SIGN": 3,
    "TYPE_CYCLIST": 4
}

camera_mapping = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT"
}

color_mapping = {
    1: 'r', 
    2: 'g',
    3: 'y',
    4: 'b'
}
def tfrecord_parser(data):
    # Create a description of the features.
    feature_description = {
        # ---------- General Frame metadata -------------
        'scene_name': tf.io.FixedLenFeature([1], tf.string),
        'time_of_day': tf.io.FixedLenFeature([1], tf.string),
        'location': tf.io.FixedLenFeature([1], tf.string),
        'weather': tf.io.FixedLenFeature([1], tf.string),
        'vehicle_pose': tf.io.FixedLenFeature([4*4], tf.float32), # 4x4 vehicle pose matrix. Basis for defining LiDAR pointCloud
        'timestamp': tf.io.FixedLenFeature([1], tf.int64),

        # ---------- NLZ (No Label Zones) ------------- # Only to be used for camera data, lidar points from NLZ are already removed
        'NLZ/num_nlz': tf.io.VarLenFeature(tf.int64),
        'NLZ/x': tf.io.VarLenFeature(tf.float32),
        'NLZ/y': tf.io.VarLenFeature(tf.float32),

        # ------ LiDAR data ------
        'LiDAR/point_cloud/num_valid_points': tf.io.FixedLenFeature([1], tf.int64),
        'LiDAR/point_cloud/xyz': tf.io.VarLenFeature(tf.float32), # *3 becoz each point has x, y, z
        'LiDAR/point_cloud/intensity': tf.io.VarLenFeature(tf.float32),
        'LiDAR/point_cloud/elongation': tf.io.VarLenFeature(tf.float32),
        'LiDAR/calibration': tf.io.FixedLenFeature([4*4], tf.float32), # Extrinsic calibration matrix for top lidar
        ## -- LiDAR Labels --
        'LiDAR/labels/num_valid_labels': tf.io.FixedLenFeature([1], tf.int64),
        'LiDAR/labels/num_lidar_points': tf.io.VarLenFeature(tf.int64),
        'LiDAR/labels/difficulty': tf.io.VarLenFeature(tf.int64),
        'LiDAR/labels/box_3d': tf.io.VarLenFeature(tf.float32), # *7 because each box has 7 values: center x, y, z, length, width, height, heading
        'LiDAR/labels/metadata': tf.io.VarLenFeature(tf.float32), # *6 becoz speed x, y, z; acceleration x, y, z
        'LiDAR/labels/class_ids': tf.io.VarLenFeature(tf.int64),
        ## -- LiDAR segmentation --
        'LiDAR/labels/segmentation/is_present': tf.io.FixedLenFeature([1], tf.int64), # 1 -> present / 0 -> absent
        'LiDAR/labels/segmentation/instance_ids': tf.io.VarLenFeature(tf.int64), # Reshape to (num_valid_points, 1)
        'LiDAR/labels/segmentation/semantic_class': tf.io.VarLenFeature(tf.int64), # Reshape to (num_valid_points, 1)
        # ------
    } 

    # Iteratively adding features for each camera
    for _cam_name in camera_mapping.values():
        _cam_name = f'Camera/{_cam_name}'
        single_camera_feature = {
            f'{_cam_name}/image': tf.io.FixedLenFeature([], tf.string),
            f'{_cam_name}/width': tf.io.FixedLenFeature([1], tf.int64),
            f'{_cam_name}/height': tf.io.FixedLenFeature([1], tf.int64),
            f'{_cam_name}/calibration/intrinsic': tf.io.FixedLenFeature([9], tf.float32),
            f'{_cam_name}/calibration/extrinsic': tf.io.FixedLenFeature([4*4], tf.float32),
            f'{_cam_name}/labels/num_valid_labels': tf.io.FixedLenFeature([1], tf.int64),
            f'{_cam_name}/labels/box_3d': tf.io.VarLenFeature(tf.float32),
            f'{_cam_name}/labels/class_ids': tf.io.VarLenFeature(tf.int64),
            f'{_cam_name}/labels/segmentation/is_present': tf.io.FixedLenFeature([1], tf.int64),
            f'{_cam_name}/labels/segmentation/panoptic_label': tf.io.FixedLenFeature([], tf.string), # tf.io.decode_png()
            f'{_cam_name}/labels/segmentation/panoptic_label_divisor': tf.io.FixedLenFeature([1], tf.int64),
            f'{_cam_name}/labels/segmentation/sequence_id': tf.io.FixedLenFeature([1], tf.string),
            f'{_cam_name}/labels/segmentation/mapping/local_instance_id': tf.io.VarLenFeature(tf.int64),
            f'{_cam_name}/labels/segmentation/mapping/global_instance_id': tf.io.VarLenFeature(tf.int64),
            f'{_cam_name}/labels/segmentation/mapping/is_tracked': tf.io.VarLenFeature(tf.int64),
        }
        feature_description.update(single_camera_feature)

    return tf.io.parse_single_example(data, feature_description)


def get_upright_3d_box_corners(boxes, name=None):
    """Given a set of upright boxes, return its 8 corners.

    Given a set of boxes, returns its 8 corners. The corners are ordered layers
    (bottom, top) first and then counter-clockwise within each layer.

    Args:
        boxes: tf Tensor [N, 7]. The inner dims are [center{x,y,z}, length, width,
        height, heading].
        name: the name scope.

    Returns:
        corners: tf Tensor [N, 8, 3].
    """
    def get_yaw_rotation(yaw, name=None):
        """Gets a rotation matrix given yaw only.

        Args:
            yaw: x-rotation in radians. This tensor can be any shape except an empty
            one.
            name: the op name.

        Returns:
            A rotation tensor with the same data type of the input. Its shape is
            [input_shape, 3 ,3].
        """

        cos_yaw = tf.cos(yaw)
        sin_yaw = tf.sin(yaw)
        ones = tf.ones_like(yaw)
        zeros = tf.zeros_like(yaw)

        return tf.stack([
            tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
            tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
            tf.stack([zeros, zeros, ones], axis=-1),
        ],
                        axis=-2)

    center_x, center_y, center_z, length, width, height, heading = tf.unstack(
        boxes, axis=-1)

    # [N, 3, 3]
    rotation = get_yaw_rotation(heading)
    # [N, 3]
    translation = tf.stack([center_x, center_y, center_z], axis=-1)

    l2 = length * 0.5
    w2 = width * 0.5
    h2 = height * 0.5

    # [N, 8, 3]
    corners = tf.reshape(
        tf.stack([
            l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
            -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2
        ],
                    axis=-1), [-1, 8, 3])
    # [N, 8, 3]
    corners = tf.einsum('nij,nkj->nki', rotation, corners) + tf.expand_dims(
        translation, axis=-2)

    return corners

def draw_3d_wireframe_box(ax, boxes, labels, linewidth=3):
    """Draws 3D wireframe bounding boxes onto the given axis."""
    # List of lines to interconnect. Allows for various forms of connectivity.
    # Four lines each describe bottom face, top face and vertical connectors.
    lines = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7))

    for box3d, label in zip(boxes, labels):    
        for (point_idx1, point_idx2) in lines:
            # line = plt.Line2D(
            #     xdata=(int(u[point_idx1]), int(u[point_idx2])),
            #     ydata=(int(v[point_idx1]), int(v[point_idx2])),
            #     linewidth=linewidth,
            #     color=list(color) + [0.5])  # Add alpha for opacity

            # ax.add_line(line)
            # print(box3d, point_idx1, point_idx2, box3d[point_idx1, 0])
            ax.scatter(
                np.linspace(int(box3d[point_idx1, 0]), int(box3d[point_idx2, 0]), 100),
                np.linspace(int(box3d[point_idx1, 1]), int(box3d[point_idx2, 1]), 100),
                np.linspace(int(box3d[point_idx1, 2]), int(box3d[point_idx2, 2]), 100), 
                s=1,
                color=color_mapping[label], linewidth=linewidth
            )

def draw_3d_bboxes(ax, bbox, classids):
    """Draws 3d rotated bboxes on the pointcloud.
    Converts 7DoF bounding box to cuboid's 8 corners.
    Creates faces (total 6 for each cuboid) for each corner, 
    and draws these faces. (Simple scatter plot doesnt work,
    it resulted in squishing/incorrect lines in different views)

    Args:
        ax (_type_): axis subplot
        bbox (np.array, shape: [num_bboxes, 7]): 3d bboxes
    """
    bboxes_8corners = get_upright_3d_box_corners(bbox).numpy()
    print(bboxes_8corners.shape)

    all_bboxes_face_connected = []
    for bbox in bboxes_8corners:
        total_box_face_connections = [
            [0, 4, 7, 3],
            [0, 1, 2, 3],
            [0, 1, 5, 4], 
            [4, 5, 6, 7],
            [2, 3, 7, 6],
            [2, 3, 6, 5]
        ]
        total_faces = []
        for face_connection in total_box_face_connections:
            face_coordinates = []
            for point_idx in face_connection:
                face_coordinates.append(bbox[point_idx])
            total_faces.append(face_coordinates)
        all_bboxes_face_connected.extend(total_faces)

    all_bboxes_face_connected = np.array(all_bboxes_face_connected)
    
    classid_to_color = [color_mapping[x] for x in classids.numpy()]
    faces_3d_collection = art3d.Poly3DCollection(
        all_bboxes_face_connected, facecolors=np.repeat(classid_to_color, 6), edgecolor="k", alpha=0.3
    )
    ax.add_collection3d(faces_3d_collection)

    
def visualize_pointcloud(decoded_lidar_data):
    fig = plt.figure(figsize=(10, 10))

    SKIP_EVERY_N_POINTS = 20
    x = decoded_lidar_data["pointcloud"][:, 0][::SKIP_EVERY_N_POINTS]
    y = decoded_lidar_data["pointcloud"][:, 1][::SKIP_EVERY_N_POINTS]
    z = decoded_lidar_data["pointcloud"][:, 2][::SKIP_EVERY_N_POINTS]

    dist = tf.math.sqrt([x**2+y**2+z**2])
    
    # --- 2D plotting ---
    # ax = fig.add_subplot()
    # ax.scatter(x, y, s=0.5, c=dist, cmap="viridis", alpha=0.6)
    # for box, classid in zip(decoded_lidar_data["box3d"].numpy(), decoded_lidar_data["classids"].numpy()):
    #     width, height = box[3], box[4]
    #     x = box[0] - width/2
    #     y = box[1] - height/2

    #     rect = patches.Rectangle(xy=(x, y), width=width, height=height, angle=box[6], rotation_point='center', linewidth=1, edgecolor=color_mapping[classid], facecolor='none')
    #     ax.add_patch(rect)
    # ax.set_xlim([-65, 65])
    # ax.set_ylim([-65, 65])
    # plt.savefig(f"output/vis_pc_bev.png", bbox_inches='tight', pad_inches=0)
    

    ax = fig.add_subplot(projection="3d") # Axes3D(fig)
    ax.set_axis_off()
    ax.scatter(x, y, z, s=0.5, c=dist, cmap="viridis", alpha=0.6)
    view_mapping = [
        # (60, 30, "FRONT_RIGHT"),
        # (60, 210, "BACK_LEFT"),
        # (60, 90, "SIDE"),
        (90, 0, "TOP")
    ]
    # Plot labels
    # box_corners_3d = get_upright_3d_box_corners(decoded_lidar_data["box3d"]).numpy()
    # print(box_corners_3d.shape)
    # for box3d, label_class_id in zip(box_corners_3d, decoded_lidar_data["classids"].numpy()):
    #     ax.add_collection3d(art3d.Poly3DCollection([box3d], edgecolors=color_mapping[label_class_id], facecolors='none', linewidths=1, alpha=0.5))
    draw_3d_bboxes(ax, decoded_lidar_data["box3d"], decoded_lidar_data["classids"])
    for elev, azim, title in view_mapping:
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d(-20, 20)
        ax.set_ylim3d(-20, 20)
        ax.set_zlim3d(0, 4)

        ax.set_title(title)
        # xlim = ax.get_xlim3d()
        # ylim = ax.get_ylim3d()
        # zlim = ax.get_zlim3d()
        # ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
        # draw_3d_wireframe_box(ax, box_corners_3d, decoded_lidar_data["classids"].numpy())
        plt.savefig(f"output/vis_pc3d_{title}.png", bbox_inches='tight', pad_inches=0)

def _decode_lidar_data(parsed_frame_data):
    return {
        "num_valid_lidar_points": parsed_frame_data['LiDAR/point_cloud/num_valid_points'],
        "pointcloud": tf.reshape(tf.sparse.to_dense(parsed_frame_data['LiDAR/point_cloud/xyz']), (-1, 3)),
        "intensity": tf.sparse.to_dense(parsed_frame_data['LiDAR/point_cloud/intensity']),
        "elongation": tf.sparse.to_dense(parsed_frame_data['LiDAR/point_cloud/elongation']),

        "num_valid_lidar_label": parsed_frame_data['LiDAR/labels/num_valid_labels'],
        "num_lidar_points_per_label": tf.sparse.to_dense(parsed_frame_data['LiDAR/labels/num_lidar_points']),
        "difficulty": tf.sparse.to_dense(parsed_frame_data['LiDAR/labels/difficulty']),
        "box3d": tf.reshape(tf.sparse.to_dense(parsed_frame_data['LiDAR/labels/box_3d']), (-1, 7)),
        "metadata": tf.reshape(tf.sparse.to_dense(parsed_frame_data['LiDAR/labels/metadata']), (-1, 6)),
        "classids": tf.sparse.to_dense(parsed_frame_data['LiDAR/labels/class_ids'])
    }
    

TF_RECORD_PATH = "../sample_waymo_write_directory/training/segment-1051897962568538022_238_170_258_170_with_camera_labels.tfrecord"
raw_dataset = tf.data.TFRecordDataset(TF_RECORD_PATH, "GZIP")
parsed_dataset = raw_dataset.map(tfrecord_parser)

for _parsed_frame in parsed_dataset.take(10):
    lidar_data = _decode_lidar_data(_parsed_frame)
    visualize_pointcloud(lidar_data)
    # print(_parsed_frame)
    # print(_parsed_frame.keys())
    break