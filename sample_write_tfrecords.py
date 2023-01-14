import tensorflow as tf
import os
import math
import numpy as np
import itertools
from multiprocessing import Pool

from waymo_open_dataset.utils import range_image_utils, box_utils, transform_utils, frame_utils
from waymo_open_dataset.camera.ops import py_camera_model_ops
from waymo_open_dataset import dataset_pb2 as open_dataset

import open3d as o3d
import random
import matplotlib.pyplot as plt
from matplotlib import patches

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

def printp(data):
    print(data, type(data))

def padToSize(data, size, pad_value=-1):
    return np.pad(data, (0, size - data.shape[0]), constant_values=pad_value)

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_array_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def project_vehicle_to_image(vehicle_pose, calibration, points):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    vehicle_pose: Vehicle pose transform from vehicle into world coordinate
      system.
    calibration: Camera calibration details (including intrinsics/extrinsics).
    points: Points to project of shape [N, 3] in vehicle coordinate system.

  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """
  # Transform points from vehicle to world coordinate system (can be
  # vectorized).
  pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ], dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

  # Perform projection and return projected image coordinates (u, v, ok).
  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()

def _process_projected_camera_synced_boxes(camera_image, frame_data, ax=False, draw_3d_box=True):
  """Displays camera_synced_box 3D labels projected onto camera."""
  def draw_3d_wireframe_box(ax, u, v, color, linewidth=3):
    """Draws 3D wireframe bounding boxes onto the given axis."""
    # List of lines to interconnect. Allows for various forms of connectivity.
    # Four lines each describe bottom face, top face and vertical connectors.
    lines = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7))

    for (point_idx1, point_idx2) in lines:
        line = plt.Line2D(
            xdata=(int(u[point_idx1]), int(u[point_idx2])),
            ydata=(int(v[point_idx1]), int(v[point_idx2])),
            linewidth=linewidth,
            color=list(color) + [0.5])  # Add alpha for opacity
        ax.add_line(line)

  # Fetch matching camera calibration.
  calibration = next(cc for cc in frame_data.context.camera_calibrations
                     if cc.name == camera_image.name)

  list_box3d = []
  list_classids = []
  
  for label in frame_data.laser_labels:
    box = label.camera_synced_box

    if not box.ByteSize():
      continue  # Filter out labels that do not have a camera_synced_box.
    FILTER_AVAILABLE = any(
        [label.num_top_lidar_points_in_box > 0 for label in frame_data.laser_labels]
    )

    if (FILTER_AVAILABLE and not label.num_top_lidar_points_in_box) or (
        not FILTER_AVAILABLE and not label.num_lidar_points_in_box):
      continue  # Filter out likely occluded objects.

    # Retrieve upright 3D box corners.
    box_coords = np.array([[
        box.center_x, box.center_y, box.center_z, box.length, box.width,
        box.height, box.heading
    ]])
    corners = box_utils.get_upright_3d_box_corners(
        box_coords)[0].numpy()  # [8, 3]

    # Project box corners from vehicle coordinates onto the image.
    projected_corners = project_vehicle_to_image(frame_data.pose, calibration,
                                                 corners)
    u, v, ok = projected_corners.transpose()
    ok = ok.astype(bool)

    # Skip object if any corner projection failed. Note that this is very
    # strict and can lead to exclusion of some partially visible objects.
    if not all(ok):
      continue
    u = u[ok]
    v = v[ok]

    # Clip box to image bounds.
    u = np.clip(u, 0, calibration.width)
    v = np.clip(v, 0, calibration.height)

    if u.max() - u.min() == 0 or v.max() - v.min() == 0:
      continue

    # if draw_3d_box:
    #   # Draw approximate 3D wireframe box onto the image. Occlusions are not
    #   # handled properly.
    #   draw_3d_wireframe_box(ax, u, v, (1.0, 1.0, 0.0))
    # else:
    #   # Draw projected 2D box onto the image.
    #   draw_2d_box(ax, u, v, (1.0, 1.0, 0.0))

    # print(u, v, np.column_stack((u, v)).reshape(8, 2).flatten())
    list_box3d.append(np.column_stack((u, v)).reshape(8, 2).flatten())
    list_classids.append(label.type)

  num_valid_labels = len(list_box3d)
  return np.array(list_box3d).flatten(), np.array(list_classids).flatten(), num_valid_labels

def convert_range_image_to_point_cloud_labels(
        frame,
        range_images,
        segmentation_labels,
        ri_index=0
    ):
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            s1 = segmentation_labels[c.name][ri_index]
            s1_tensor = tf.reshape(tf.convert_to_tensor(s1.data), s1.shape.dims)
            s1_points_tensor = tf.gather_nd(s1_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            s1_points_tensor = tf.zeros([num_valid_point, 2], tf.int32)

        point_labels.append(s1_points_tensor.numpy())
    return point_labels

def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
  """Convert range images to point cloud.
  FUNCTION is exactly the same as from waymo_open_dataset but ignores points
  from NLZ

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  points = []
  cp_points = []

  cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
      frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_NLZ_mask = range_image_tensor[..., -1] > 0 # Ignore NLZ
    range_image_final_mask = tf.math.logical_and(range_image_mask, range_image_NLZ_mask)

    range_image_cartesian = cartesian_range_images[c.name]
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.compat.v1.where(range_image_final_mask))

    # cp = camera_projections[c.name][ri_index]
    # cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
    # cp_points_tensor = tf.gather_nd(cp_tensor,
    #                                 tf.compat.v1.where(range_image_final_mask))
    points.append(points_tensor.numpy())
    # cp_points.append(cp_points_tensor.numpy())

  return points #, cp_points


def _process_camera_data(frame_data):  
    # plt.figure(figsize=(25, 20))
    final_dict = {}
    for index, cam_img in enumerate(frame_data.images):
        camera_name = camera_mapping[cam_img.name] # str
        img_data = cam_img.image # Already encoded. Decode to jpeg using: tf.image.decode_jpeg()
        
        calibration = [cc for cc in frame_data.context.camera_calibrations if cc.name == cam_img.name][0]
        intrinsic_mtx =  np.array(calibration.intrinsic) # float 1d array: 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]., k: radial, p: Tangential distortions
        extrinsic_mtx = np.array(calibration.extrinsic.transform) # float 4x4
        img_width = calibration.width # int
        img_height = calibration.height # int

        box3d_arr, class_ids_arr, num_valid = _process_projected_camera_synced_boxes(cam_img, frame_data) # float [N, 16] for boxes (16: xyxy... for 8 points), int [N] for class_ids
        class_ids_arr = class_ids_arr.astype(int)

        _cam_name = f'Camera/{camera_name}'
        single_camera_feature = {
            f'{_cam_name}/image': _bytes_feature(img_data),
            f'{_cam_name}/width': _int64_feature(img_width),
            f'{_cam_name}/height': _int64_feature(img_height),
            f'{_cam_name}/calibration/intrinsic': _float_array_feature(intrinsic_mtx.tolist()),
            f'{_cam_name}/calibration/extrinsic': _float_array_feature(extrinsic_mtx.tolist()),
            f'{_cam_name}/labels/num_valid_labels': _int64_feature(num_valid),
            f'{_cam_name}/labels/box_3d': _float_array_feature(box3d_arr.tolist()),
            f'{_cam_name}/labels/class_ids': _int64_array_feature(class_ids_arr.tolist())
        }

        if cam_img.camera_segmentation_label.panoptic_label:
            seg_key = cam_img.camera_segmentation_label
            panoptic_label = seg_key.panoptic_label # bytes -> tf.io.decode_png
            panoptic_label_divisor = seg_key.panoptic_label_divisor # int
            sequence_id = seg_key.sequence_id # string
            
            map_local_instance_list = []
            map_global_instance_list = []
            map_is_tracked_list = []
            for mapping in seg_key.instance_id_to_global_id_mapping:
                map_local_instance_list.append(mapping.local_instance_id) # int
                map_global_instance_list.append(mapping.global_instance_id) # int
                map_is_tracked_list.append(mapping.is_tracked) # bool

            map_local_instance_list = np.array(map_local_instance_list).flatten()
            map_global_instance_list = np.array(map_global_instance_list).flatten()
            map_is_tracked_list = np.array(map_is_tracked_list, np.int32).flatten()

            cam_segmentation = {
                f'{_cam_name}/labels/segmentation/is_present': _int64_feature(1),
                f'{_cam_name}/labels/segmentation/panoptic_label': _bytes_feature(panoptic_label), # tf.io.decode_png()
                f'{_cam_name}/labels/segmentation/panoptic_label_divisor': _int64_feature(panoptic_label_divisor),
                f'{_cam_name}/labels/segmentation/sequence_id': _bytes_feature(sequence_id.encode('utf-8')),
                f'{_cam_name}/labels/segmentation/mapping/local_instance_id': _int64_array_feature(map_local_instance_list.tolist()),
                f'{_cam_name}/labels/segmentation/mapping/global_instance_id': _int64_array_feature(map_global_instance_list.tolist()),
                f'{_cam_name}/labels/segmentation/mapping/is_tracked': _int64_array_feature(map_is_tracked_list.tolist()),
            }
        else:
            # Added invalid data for feature-key consistency
            cam_segmentation = {
                f'{_cam_name}/labels/segmentation/is_present': _int64_feature(0),
                f'{_cam_name}/labels/segmentation/panoptic_label': _bytes_feature(tf.io.serialize_tensor(np.zeros((2, 2, 1)))), # tf.io.decode_png()
                f'{_cam_name}/labels/segmentation/panoptic_label_divisor': _int64_feature(0),
                f'{_cam_name}/labels/segmentation/sequence_id': _bytes_feature("null".encode('utf-8')),
                f'{_cam_name}/labels/segmentation/mapping/local_instance_id': _int64_array_feature(np.full((2), -1).flatten().tolist()),
                f'{_cam_name}/labels/segmentation/mapping/global_instance_id': _int64_array_feature(np.full((2), -1).flatten().tolist()),
                f'{_cam_name}/labels/segmentation/mapping/is_tracked': _int64_array_feature(np.full((2), -1).flatten().tolist()),
            }
        
        camera_dict = {**single_camera_feature, **cam_segmentation}
        final_dict.update(camera_dict)

    return final_dict

def _process_lidar_labels(frame_data):
    """Creates lidar label feature dict.

    Args:
        frame_data (_type_): Parsed frame data
    """
    
    list_box3d = []
    list_meta = []
    list_num_lidar_points = []
    list_class_ids = []
    list_difficulty = []

    for label in frame_data.laser_labels:
        # printp(label)
        list_difficulty.append(label.detection_difficulty_level) # int
        box = label.box
        box_coords = np.array([[
            box.center_x, box.center_y, box.center_z, box.length, box.width,
            box.height, box.heading
        ]])
        list_box3d.append(box_coords) # float32

        meta = label.metadata
        meta_values = np.array([meta.speed_x , meta.speed_y, meta.accel_x, meta.accel_y, meta.speed_z, meta.accel_z])
        list_meta.append(meta_values) # float32

        num_vis_points = label.num_lidar_points_in_box 
        list_num_lidar_points.append(num_vis_points) # int

        list_class_ids.append(label.type) # int

    # --- Flatten and pad data to constant size --- | In labels: 250
    num_valid_labels = len(list_box3d)
    difficulty = np.array(list_class_ids).flatten()
    box3d = np.array(list_box3d).flatten()
    meta = np.array(list_meta).flatten()
    num_lidar_points = np.array(list_num_lidar_points).flatten()
    class_ids = np.array(list_class_ids).flatten()
    # ---

    box_label_feature_dict = {
        'LiDAR/labels/num_valid_labels': _int64_feature(num_valid_labels),
        'LiDAR/labels/num_lidar_points': _int64_array_feature(num_lidar_points.tolist()),
        'LiDAR/labels/difficulty': _int64_array_feature(difficulty.tolist()),
        'LiDAR/labels/box_3d': _float_array_feature(box3d.tolist()),
        'LiDAR/labels/metadata': _float_array_feature(meta.tolist()),
        'LiDAR/labels/class_ids': _int64_array_feature(class_ids.tolist())
    }

    return box_label_feature_dict

def _process_lidar_pointcloud(frame_data):
    # Read range images and corresponding projections from waymo dataset
    (range_images, camera_projections, 
    seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame_data)
    # Convert range images to proper cartesian point cloud and combine all lidar sensor data to one
    cartesian_pc_rg0 = convert_range_image_to_point_cloud(frame_data, range_images, camera_projections, range_image_top_pose, 0, True)
    cartesian_pc_rg0 = np.concatenate(cartesian_pc_rg0, axis=0) # LiDAR return 0

    cartesian_pc_rg1 = convert_range_image_to_point_cloud(frame_data, range_images, camera_projections, range_image_top_pose, 0, True)
    cartesian_pc_rg1 = np.concatenate(cartesian_pc_rg1, axis=0) # LiDAR return 1

    cartesian_pc = np.concatenate((cartesian_pc_rg0, cartesian_pc_rg1), axis=0) # Combines both the LiDAR returns into one pc

    # --- Flatten --- 
    point_cloud_xyz = cartesian_pc[:, 3:]
    valid_points = point_cloud_xyz.shape[0]  
    point_cloud_intensity = cartesian_pc[:, 1].flatten().tolist()
    point_cloud_elongation = cartesian_pc[:, 2].flatten().tolist()
    # ---

    # --- Process lidar segmentation data -- (Not guranteed to be present for each frame)
    is_seg_label_present = 0
    if seg_labels:
        is_seg_label_present = 1
        point_labels_r0 = np.concatenate(convert_range_image_to_point_cloud_labels(frame_data, range_images, seg_labels), axis=0)
        point_labels_r1 = np.concatenate(convert_range_image_to_point_cloud_labels(frame_data, range_images, seg_labels, 1), axis=0)
        point_labels = np.concatenate((point_labels_r0, point_labels_r1), axis=0) # instance_id, semantic_class
        instance_id = point_labels[..., 0].flatten().tolist() # int
        semantic_class = point_labels[..., 1].flatten().tolist() # int
    else:
        instance_id = np.full((2), -1).flatten().tolist()
        semantic_class = np.full((2), -1).flatten().tolist()
    # ---

    # Get extrinsic calibration for TOP LiDAR
    for laser_calib in frame_data.context.laser_calibrations:
        if laser_calib.name == 1:
            extrinsic_calib = np.array(laser_calib.extrinsic.transform).tolist()
            break

    pointcloud_feature_dict = {
        'LiDAR/point_cloud/num_valid_points': _int64_feature(valid_points),
        'LiDAR/point_cloud/xyz': _float_array_feature(point_cloud_xyz.flatten().tolist()),
        'LiDAR/point_cloud/intensity': _float_array_feature(point_cloud_intensity),
        'LiDAR/point_cloud/elongation': _float_array_feature(point_cloud_elongation),
        'LiDAR/calibration': _float_array_feature(extrinsic_calib),
        'LiDAR/labels/segmentation/is_present': _int64_feature(is_seg_label_present),
        'LiDAR/labels/segmentation/instance_ids': _int64_array_feature(instance_id),
        'LiDAR/labels/segmentation/semantic_class': _int64_array_feature(semantic_class)
    }

    return pointcloud_feature_dict

def _process_nlz(frame_data):
    # No label zones
    list_x = []
    list_y = []
    list_length = []

    nlz = frame_data.no_label_zones
    if len(nlz) > 0:
        for _nlz in nlz:
            list_x.extend(_nlz.x)
            list_y.extend(_nlz.y)
            list_length.append(len(_nlz.x))
    else:
        list_x.append(-1.0)
        list_y.append(-1.0)
        list_length.append(0)

    nlz_feature = {
        "NLZ/num_nlz": _int64_array_feature(list_length),
        "NLZ/x": _float_array_feature(list_x),
        "NLZ/y": _float_array_feature(list_y)
    }

    return nlz_feature
    
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


def serialize_sample(frame): #, pc_num_points, point_cloud_data, pc_intensity, pc_elongation):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    frame_meta_feature = {
        'scene_name': _bytes_feature(frame.context.name.encode('utf-8')),
        'time_of_day': _bytes_feature(frame.context.stats.time_of_day.encode('utf-8')),
        'location': _bytes_feature(frame.context.stats.location.encode('utf-8')),
        'weather': _bytes_feature(frame.context.stats.weather.encode('utf-8')),
        'vehicle_pose': _float_array_feature(np.array(frame.pose.transform).tolist()),
        'timestamp': _int64_feature(frame.timestamp_micros),
    }

    # Combining all individual feature dicts into one
    final_feature = {
        **frame_meta_feature, 
        **_process_nlz(frame),
        **_process_lidar_pointcloud(frame), 
        **_process_lidar_labels(frame), 
        **_process_camera_data(frame)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=final_feature))
    return example_proto.SerializeToString()


def process_single_segment(paths):
    segment_path, destination_path = paths
    tf_record = tf.data.TFRecordDataset(segment_path, compression_type="")

    with tf.io.TFRecordWriter(destination_path, tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
        for index, data in enumerate(tf_record):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            converted_frame = serialize_sample(frame)
            writer.write(converted_frame)
            print(f"File: {segment_path}, Processed {index} frame")

            # if index == 4:
            #     break

tf.config.set_visible_devices([], "GPU")
source_folder = "/mnt/d/datasets/waymo/"
destination_folder = "../sample_waymo_write_directory"

for subfolder in ["training", "testing", "validation"]:
    source_sub_folder = os.path.join(source_folder, subfolder)
    destination_sub_folder = os.path.join(destination_folder, subfolder)
    os.makedirs(destination_sub_folder, exist_ok=True)
    
    tfrecord_files = os.listdir(source_sub_folder)

    source_tfrecord_files = [os.path.join(source_sub_folder, tfrecord_file) for tfrecord_file in tfrecord_files]
    destination_tfrecord_files = [os.path.join(destination_sub_folder, tfrecord_file) for tfrecord_file in tfrecord_files]

    with Pool() as pool:
        pool.map(process_single_segment, zip(source_tfrecord_files, destination_tfrecord_files))

# Test read dataset
raw_dataset = tf.data.TFRecordDataset("../sample_waymo_write_directory/testing/segment-10149575340910243572_2720_000_2740_000_with_camera_labels.tfrecord", "GZIP")
parsed_dataset = raw_dataset.map(tfrecord_parser)

for _parsed_record in parsed_dataset.take(10):
    print(_parsed_record.keys())
    break