import h5py

import Utils.bvh as bvh
from Utils import quat
from scipy.interpolate import griddata
import scipy.signal as signal
import numpy as np
import pandas as pd
import os

np.set_printoptions(precision=6, suppress=True)
window = 60


def animation_mirror(lrot, lpos, names, parents):
    joints_mirror = np.array([(
        names.index('Left' + n[5:]) if n.startswith('Right') else (
            names.index('Right' + n[4:]) if n.startswith('Left') else
            names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    grot, gpos = quat.fk(lrot, lpos, parents)

    gpos_mirror = mirror_pos * gpos[:, joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:, joints_mirror]))

    return quat.ik(grot_mirror, gpos_mirror, parents)


def data_process(rotations, positions, parents, style, selected):
    # Compute velocities via central difference
    velocities = np.empty_like(positions)
    velocities[1:-1] = (
            0.5 * (positions[2:] - positions[1:-1]) * 60.0 +
            0.5 * (positions[1:-1] - positions[:-2]) * 60.0)
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

    # Same for angular velocities
    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.0 +
            0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.0)
    angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

    global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
        rotations, positions, velocities, angular_velocities, parents)

    # direction comes from projected hip forward direction
    direction = np.array([1, 0, 1]) * quat.mul_vec(
        global_rotations[:, 0:1], np.array([0.0, 0.0, 1.0])
    )

    # smooth and re-normalize
    direction = direction / np.sqrt(np.sum(np.square(direction), axis=-1))[..., np.newaxis]
    direction = signal.savgol_filter(direction, 61, 3, axis=0, mode='interp')
    direction = direction / np.sqrt(np.sum(np.square(direction), axis=-1))[..., np.newaxis]

    # root rotations
    root_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), direction))

    # root velocity
    root_velocity = np.array([1, 0, 1]) * quat.inv_mul_vec(
        root_rotation, global_velocities[:, 0:1]
    )

    # root angular velocity
    root_angular_velocity = np.degrees(quat.inv_mul_vec(root_rotation, global_angular_velocities[:, 0:1]))

    # local space
    local_positions = global_positions.copy()
    local_positions[:, :, 0] = local_positions[:, :, 0] - local_positions[:, 0:1, 0]
    local_positions[:, :, 2] = local_positions[:, :, 2] - local_positions[:, 0:1, 2]
    local_positions = quat.inv_mul_vec(root_rotation, local_positions)

    local_rotations = quat.inv_mul(root_rotation, global_rotations)
    local_xy = quat.to_xform_xy(local_rotations)
    local_velocities = quat.inv_mul_vec(root_rotation, global_velocities)

    # build up database
    X, Y = [], []

    for i in range(window, len(positions) - window - 1, 1):
        root_position = np.array([1, 0, 1]) * quat.inv_mul_vec(
            root_rotation[i:i + 1], global_positions[i - window:i + window:10, 0:1] - global_positions[i:i + 1, 0:1])
        root_direction = quat.inv_mul_vec(root_rotation[i:i + 1], direction[i - window:i + window:10])
        root_vel = quat.inv_mul_vec(root_rotation[i:i + 1], global_velocities[i - window:i + window:10, 0:1])
        speed = np.sqrt(np.sum(np.square(root_vel), axis=-1))

        X.append(np.hstack([
            root_position[:, :, 0].ravel(), root_position[:, :, 2].ravel(),
            root_direction[:, :, 0].ravel(), root_direction[:, :, 2].ravel(),
            root_vel[:, :, 0].ravel(), root_vel[:, :, 2].ravel(),
            speed.ravel(),

            local_positions[i - 1, selected].ravel(),
            local_xy[i - 1, selected, :, 0].ravel(), local_xy[i - 1, selected, :, 1].ravel(),
            local_velocities[i - 1, selected].ravel(),
        ]))

        root_position_next = np.array([1, 0, 1]) * quat.inv_mul_vec(
            root_rotation[i + 1:i + 2],
            global_positions[i + 1:i + window + 1:10, 0:1] - global_positions[i + 1:i + 2, 0:1])
        root_direction_next = quat.inv_mul_vec(root_rotation[i + 1:i + 2], direction[i + 1:i + window + 1:10])
        root_vel_next = quat.inv_mul_vec(root_rotation[i + 1:i + 2], global_velocities[i + 1:i + window + 1:10, 0:1])

        Y.append(np.hstack([
            root_velocity[i, :, 0].ravel(), root_angular_velocity[i, :, 1].ravel(), root_velocity[i, :, 2].ravel(),

            root_position_next[:, :, 0].ravel(), root_position_next[:, :, 2].ravel(),
            root_direction_next[:, :, 0].ravel(), root_direction_next[:, :, 2].ravel(),
            root_vel_next[:, :, 0].ravel(), root_vel_next[:, :, 2].ravel(),

            local_positions[i, selected].ravel(),
            local_xy[i, selected, :, 0].ravel(), local_xy[i, selected, :, 1].ravel(),
            local_velocities[i, selected].ravel()
        ]))

    return np.array(X), np.array(Y)


def load_frame_cuts(csv_path='./100STYLE/Frame_Cuts.csv',
                    base_dir='./100STYLE',
                    suffixes=None):
    """
    读取包含帧切割信息的CSV文件并生成BVH文件路径及帧范围列表

    参数：
    csv_path: str - 包含帧范围信息的CSV文件路径
    base_dir: str - 动作文件存储的基础目录
    suffixes: list[str] - 需要处理的动作后缀列表，默认为['BR','BW','FR','FW','ID','SR','SW','TR1']

    返回：
    list[tuple] - 包含（BVH文件完整路径，起始帧，结束帧，动作名称）的元组列表
    """
    # 设置默认后缀列表
    if suffixes is None:
        suffixes = ['BR', 'BW', 'FR', 'FW', 'ID', 'SR', 'SW', 'TR1']

    # 读取CSV数据
    data = pd.read_csv(csv_path)
    formatted_entries = []

    # 处理每一行数据
    for _, row in data.iterrows():
        style_name = row['STYLE_NAME']
        style_dir = os.path.join(base_dir, style_name).replace('\\', '/')

        # 处理每个动作后缀
        for suffix in suffixes:
            start_col = f"{suffix}_START"
            stop_col = f"{suffix}_STOP"

            # 验证列和数据有效性
            if all(col in row for col in [start_col, stop_col]):
                start = row[start_col]
                stop = row[stop_col]

                if pd.notna(start) and pd.notna(stop):
                    # 构建完整文件路径
                    bvh_file = f"{style_name}_{suffix}.bvh"
                    full_path = os.path.join(style_dir, bvh_file).replace('\\', '/')

                    # 添加到结果列表
                    formatted_entries.append((
                        full_path,
                        int(start),
                        int(stop),
                        style_name
                    ))

    return formatted_entries


files = load_frame_cuts(csv_path='E:/资料&文件/DATASET/100STYLE_re/Frame_Cuts.csv',
                        base_dir='E:/资料&文件/DATASET/100STYLE_re'
                        )[:8 * 3]

# 获取唯一风格类别并创建映射
unique_styles = sorted(list(set([file[3] for file in files])))
style_to_label = {style: idx for idx, style in enumerate(unique_styles)}
num_classes = len(unique_styles)  # 新增：获取类别数量
print("Style Labels:", style_to_label)
print("Number of classes:", num_classes)

with h5py.File('./100STYLE_re_label_camdm.hdf5', 'w') as h5f:
    input_dset = None
    output_dset = None
    label_dset = None  # 新增：用于存储标签的dataset

    for filename, start, stop, style in files:
        current_label = style_to_label[style]  # 获取当前style标签
        for mirror in [False, True]:
            print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))

            bvh_data = bvh.load_zeroeggs(filename)
            parents = bvh_data['parents']
            bvh_data['positions'] = bvh_data['positions'][start:stop]
            bvh_data['rotations'] = bvh_data['rotations'][start:stop]

            positions = bvh_data['positions']
            rotations = quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order'])
            rotations = quat.unroll(rotations)

            positions *= 0.01

            if mirror:
                rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
                rotations = quat.unroll(rotations)

            nframes = positions.shape[0]
            print('frames: ', nframes)
            nbones = positions.shape[1]

            rotations = quat.normalize(rotations)

            selected = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71]

            x, y = data_process(rotations, positions, parents, style, selected)

            # 创建标签数组（每个时间步使用相同的标签）
            labels = np.full((x.shape[0], 1), current_label, dtype=np.int32)

            if input_dset is None:
                input_dset = h5f.create_dataset(
                    'X',
                    shape=(0, *x.shape[1:]),
                    maxshape=(None, *x.shape[1:]),
                    dtype=np.float32,
                    compression='gzip',
                    chunks=True
                )
            input_dset.resize(input_dset.shape[0] + x.shape[0], axis=0)
            input_dset[-x.shape[0]:] = x

            if output_dset is None:
                output_dset = h5f.create_dataset(
                    'Y',
                    shape=(0, *y.shape[1:]),
                    maxshape=(None, *y.shape[1:]),
                    dtype=np.float32,
                    compression='gzip',
                    chunks=True
                )
            output_dset.resize(output_dset.shape[0] + y.shape[0], axis=0)
            output_dset[-y.shape[0]:] = y

            if label_dset is None:
                label_dset = h5f.create_dataset(
                    'Labels',
                    shape=(0, 1),
                    maxshape=(None, 1),
                    dtype=np.int32,
                    compression='gzip',
                    chunks=True
                )
            label_dset.resize(label_dset.shape[0] + labels.shape[0], axis=0)
            label_dset[-labels.shape[0]:] = labels

        print(input_dset.shape)
        print(output_dset.shape)
        print(label_dset.shape)

