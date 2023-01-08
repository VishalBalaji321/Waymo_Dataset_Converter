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
    difficulty = padToSize(np.array(list_class_ids).flatten(), pad_size)
    box3d = padToSize(np.array(list_box3d).flatten(), pad_size * 7)
    meta = padToSize(np.array(list_meta).flatten(), pad_size * 6)
    num_lidar_points = padToSize(np.array(list_num_lidar_points).flatten(), pad_size)
    class_ids = padToSize(np.array(list_class_ids).flatten(), pad_size)
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
    point_cloud_xyz_padded = padToSize(point_cloud_xyz, pad_size * 3)
    
    point_cloud_intensity = cartesian_pc[:, 1].flatten()
    point_cloud_intensity_padded = padToSize(point_cloud_intensity, pad_size)

    point_cloud_elongation = cartesian_pc[:, 2].flatten()
    point_cloud_elongation_padded = padToSize(point_cloud_elongation, pad_size)
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
        'LiDAR/point_cloud/xyz': tf.io.FixedLenFeature([250000*3], tf.float32), # *3 becoz each point has x, y, z
        'LiDAR/point_cloud/intensity': tf.io.FixedLenFeature([250000], tf.float32),
        'LiDAR/point_cloud/elongation': tf.io.FixedLenFeature([250000], tf.float32),
        'LiDAR/point_cloud/elongation': tf.io.FixedLenFeature([250000], tf.float32),
        'LiDAR/calibration': tf.io.FixedLenFeature([4*4], tf.float32), # Extrinsic calibration matrix for top lidar
        ## -- LiDAR Labels data -> Padded to 250 --
        'LiDAR/labels/num_valid_labels': tf.io.FixedLenFeature([1], tf.int64),
        'LiDAR/labels/num_lidar_points': tf.io.FixedLenFeature([250], tf.int64),
        'LiDAR/labels/difficulty': tf.io.FixedLenFeature([250], tf.int64),
        'LiDAR/labels/box_3d': tf.io.FixedLenFeature([250*7], tf.float32), # *7 because each box has 7 values: center x, y, z, length, width, height, heading
        'LiDAR/labels/metadata': tf.io.FixedLenFeature([250*6], tf.float32), # *6 becoz speed x, y, z; acceleration x, y, z
        'LiDAR/labels/class_ids': tf.io.FixedLenFeature([250], tf.int64),
        # ------
    } 

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

  final_feature = {**frame_meta_feature, **_process_lidar_pointcloud(frame), **_process_lidar_labels(frame)}
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
        example = serialize_sample(frame)#, valid_points, point_cloud_xyz_padded, point_cloud_intensity_padded, point_cloud_elongation_padded) # erframe.context.name, frame.context.stats.weather, frame.timestamp_micros)
        writer.write(example)
        
        if index == 5:
            break
    
print("Done writing to TFRecord")

# Read dataset
raw_dataset = tf.data.TFRecordDataset("sample_record.tfrecord")
parsed_dataset = raw_dataset.map(tfrecord_parser)

for _parsed_record in parsed_dataset.take(10):
    print(repr(_parsed_record))