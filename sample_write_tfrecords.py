import tensorflow as tf
import os
import math
import numpy as np
import itertools

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

def _process_projected_camera_synced_boxes(camera_image, ax=False, draw_3d_box=True):
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
  calibration = next(cc for cc in frame.context.camera_calibrations
                     if cc.name == camera_image.name)

  list_box3d = []
  list_classids = []
  
  for label in frame.laser_labels:
    box = label.camera_synced_box

    if not box.ByteSize():
      continue  # Filter out labels that do not have a camera_synced_box.
    FILTER_AVAILABLE = any(
        [label.num_top_lidar_points_in_box > 0 for label in frame.laser_labels]
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
    projected_corners = project_vehicle_to_image(frame.pose, calibration,
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

        # ax = show_camera_image(cam_img, [3, 3, index + 1])
        box3d_arr, class_ids_arr, num_valid = _process_projected_camera_synced_boxes(cam_img) # float [N, 16] for boxes (16: xyxy... for 8 points), int [N] for class_ids
        # box3d_arr = padToSize(box3d_arr, 250*8*2) # 8 corners of 3d cube, with corners xyxy
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
            # print(len(map_local_instance_list), len(map_global_instance_list), len(map_is_tracked_list))
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

def _process_lidar_labels(frame_data, pad_size=250):
    """Creates lidar label feature dict.

    Args:
        frame_data (_type_): _description_
        pad_size (int, optional): _description_. Defaults to 250.
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

def _process_lidar_pointcloud(frame_data, pad_size=250000):
    # Read range images and corresponding projections from waymo dataset
    (range_images, camera_projections, 
    _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame_data)
    # Convert range images to proper cartesian point cloud and combine all lidar sensor data to one
    cartesian_pc, cp_points = frame_utils.convert_range_image_to_point_cloud(frame_data, range_images, camera_projections, range_image_top_pose, 0, True)
    cartesian_pc = np.concatenate(cartesian_pc, axis=0) #[:, 3:].flatten()

    # --- Flatten and pad data to constant size --- | In pointcloud: 250000
    point_cloud_xyz = cartesian_pc[:, 3:].flatten()
    valid_points = point_cloud_xyz.shape[0]  
    point_cloud_intensity = cartesian_pc[:, 1].flatten()
    point_cloud_elongation = cartesian_pc[:, 2].flatten()

    # point_cloud_xyz_padded = padToSize(point_cloud_xyz, pad_size * 3)
    # point_cloud_intensity_padded = padToSize(point_cloud_intensity, pad_size)
    # point_cloud_elongation_padded = padToSize(point_cloud_elongation, pad_size)
    
    point_cloud_xyz_padded = point_cloud_xyz
    point_cloud_intensity_padded = point_cloud_intensity
    point_cloud_elongation_padded = point_cloud_elongation
    # ---

    # Get extrinsic calibration for TOP LiDAR
    for laser_calib in frame.context.laser_calibrations:
        if laser_calib.name == 1:
            extrinsic_calib = np.array(laser_calib.extrinsic.transform)

    pointcloud_feature_dict = {
        'LiDAR/point_cloud/num_valid_points': _int64_feature(valid_points),
        'LiDAR/point_cloud/xyz': _float_array_feature(point_cloud_xyz_padded.tolist()),
        'LiDAR/point_cloud/intensity': _float_array_feature(point_cloud_intensity_padded.tolist()),
        'LiDAR/point_cloud/elongation': _float_array_feature(point_cloud_elongation_padded.tolist()),
        'LiDAR/calibration': _float_array_feature(extrinsic_calib.tolist())
    }

    return pointcloud_feature_dict

def tfrecord_parser(data):
    # Create a description of the features.
    feature_description = {
        'scene_name': tf.io.FixedLenFeature([1], tf.string),
        'time_of_day': tf.io.FixedLenFeature([1], tf.string),
        'location': tf.io.FixedLenFeature([1], tf.string),
        'weather': tf.io.FixedLenFeature([1], tf.string),
        'vehicle_pose': tf.io.FixedLenFeature([4*4], tf.float32), # 4x4 vehicle pose matrix. Basis for defining LiDAR pointCloud
        'timestamp': tf.io.FixedLenFeature([1], tf.int64),
        # ------ LiDAR data ------
        'LiDAR/point_cloud/num_valid_points': tf.io.FixedLenFeature([1], tf.int64),
        ## -- Constant padded to 250000 points --
        'LiDAR/point_cloud/xyz': tf.io.VarLenFeature(tf.float32), # *3 becoz each point has x, y, z
        'LiDAR/point_cloud/intensity': tf.io.VarLenFeature(tf.float32),
        'LiDAR/point_cloud/elongation': tf.io.VarLenFeature(tf.float32),
        'LiDAR/calibration': tf.io.FixedLenFeature([4*4], tf.float32), # Extrinsic calibration matrix for top lidar
        ## -- LiDAR Labels data -> Padded to 250 --
        'LiDAR/labels/num_valid_labels': tf.io.FixedLenFeature([1], tf.int64),
        'LiDAR/labels/num_lidar_points': tf.io.VarLenFeature(tf.int64),
        'LiDAR/labels/difficulty': tf.io.VarLenFeature(tf.int64),
        'LiDAR/labels/box_3d': tf.io.VarLenFeature(tf.float32), # *7 because each box has 7 values: center x, y, z, length, width, height, heading
        'LiDAR/labels/metadata': tf.io.VarLenFeature(tf.float32), # *6 becoz speed x, y, z; acceleration x, y, z
        'LiDAR/labels/class_ids': tf.io.VarLenFeature(tf.int64)
        # ------
    } 

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
    # print(feature_description.keys())
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

  final_feature = {
    **frame_meta_feature, 
    **_process_lidar_pointcloud(frame), 
    **_process_lidar_labels(frame), 
    **_process_camera_data(frame)
  }
  print(final_feature.keys())
  print("Done encoding features")
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=final_feature))
  return example_proto.SerializeToString()

folder_path = "/mnt/d/datasets/waymo/training/"
file_path = os.path.join(folder_path, random.choice(os.listdir(folder_path)))
#file_path = os.path.join(folder_path, "segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord")

tf_record = tf.data.TFRecordDataset(file_path, compression_type="")
tf_record = tf_record.shuffle(32)

# cnt = tf_record.reduce(np.int64(0), lambda x, _: x + 1) -> Produces 199
with tf.io.TFRecordWriter("sample_record.tfrecord") as writer:
    for index, data in enumerate(tf_record):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # printp(frame.pose.transform)
        # exit()
        # print(dir(frame))
        # print(frame.context)
        # printp(frame.context.stats.weather)
        # printp(frame.timestamp_micros)
        # printp(frame.context.name)

        ## Labels
        # for label in frame.laser_labels:
        #   label.num_top_lidar_points_in_box
        # 
        example = serialize_sample(frame) #, valid_points, point_cloud_xyz_padded, point_cloud_intensity_padded, point_cloud_elongation_padded) # erframe.context.name, frame.context.stats.weather, frame.timestamp_micros)
        writer.write(example)
        
        if index == 2:
            break
    
print("Done writing to TFRecord")

# Read dataset
raw_dataset = tf.data.TFRecordDataset("sample_record.tfrecord")
parsed_dataset = raw_dataset.map(tfrecord_parser)

for _parsed_record in parsed_dataset.take(10):
    print(_parsed_record.keys())