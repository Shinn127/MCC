import Utils.bvh as bvh
from Utils import quat
import numpy as np
from scipy.interpolate import griddata
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import h5py
import pandas as pd
import os

np.set_printoptions(precision=6, suppress=True)
window = 60


def butterworth_filter(data, cutoff, fs, order=2):
    """使用 Scipy 实现的 Butterworth 滤波器"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data.astype(np.float32)).astype(np.float32)


def animation_mirror(lrot, lpos, names, parents):
    joints_mirror = np.array([
        names.index('Left' + n[5:]) if n.startswith('Right') else
        names.index('Right' + n[4:]) if n.startswith('Left') else
        names.index(n) for n in names
    ])

    mirror_pos = np.array([-1, 1, 1], dtype=np.float32)
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]], dtype=np.float32)

    grot, gpos = quat.fk(lrot, lpos, parents)
    gpos_mirror = mirror_pos * gpos[:, joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:, joints_mirror]))
    return quat.ik(grot_mirror, gpos_mirror, parents)


def data_process(rotations, positions, parents, selected):
    # 转换数据类型为 float32
    positions = positions.astype(np.float32)
    rotations = rotations.astype(np.float32)

    # 计算速度
    velocities = np.empty_like(positions, dtype=np.float32)
    velocities[1:-1] = 0.5 * (positions[2:] - positions[:-2]) * 60.0
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

    # 计算角速度
    angular_velocities = np.zeros_like(positions, dtype=np.float32)
    angular_velocities[1:-1] = 0.5 * (
            quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) +
            quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2])))
    ) * 60.0
    angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

    # 正向运动学
    global_rot, global_pos = quat.fk(rotations, positions, parents)
    del rotations, positions  # 及时释放内存

    # 计算方向特征
    direction = np.array([1, 0, 1], dtype=np.float32) * quat.mul_vec(global_rot[:, 0:1],
                                                                     np.array([0.0, 0.0, 1.0], dtype=np.float32))
    direction /= np.sqrt(np.sum(np.square(direction), axis=-1, keepdims=True)) + 1e-8
    direction = signal.savgol_filter(direction, 61, 3, axis=0, mode='interp')
    direction /= np.sqrt(np.sum(np.square(direction), axis=-1, keepdims=True)) + 1e-8

    # 根节点特征
    root_rotation = quat.normalize(quat.between(np.array([0, 0, 1], dtype=np.float32), direction))
    root_velocity = np.array([1, 0, 1], dtype=np.float32) * quat.inv_mul_vec(root_rotation, velocities[:, 0:1])
    local_positions = quat.inv_mul_vec(root_rotation, global_pos - global_pos[:, 0:1])
    local_velocities = quat.inv_mul_vec(root_rotation, velocities)
    del global_pos, velocities  # 释放不再使用的内存

    # 窗口归一化
    padded = np.pad(local_velocities, ((window, window), (0, 0), (0, 0)), mode='edge')
    normalized = np.zeros_like(local_velocities)
    for i in range(len(local_velocities)):
        window_data = padded[i:i + 2 * window + 1]
        normalized[i] = (local_velocities[i] - np.mean(window_data, axis=0))  # 依照原论文实现，不除标准差

    # 滤波器应用（优化后的向量化实现）
    n_samples, n_joints, n_axes = normalized.shape
    filtered = np.zeros_like(normalized)
    for j in range(n_joints):
        for k in range(n_axes):
            filtered[:, j, k] = butterworth_filter(normalized[:, j, k], cutoff=3.25, fs=60.0)
    del normalized  # 释放中间变量

    # print(filtered.shape)
    return np.array(filtered[:, selected], dtype=np.float32)


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


if __name__ == '__main__':
    files = load_frame_cuts(csv_path='/home/adm1n/桌面/myw/My_Character_Control/Dataset/100STYLE_re/Frame_Cuts.csv',
                            base_dir='/home/adm1n/桌面/myw/My_Character_Control/Dataset/100STYLE_re'
                            )

    unique_styles = sorted(list(set([file[3] for file in files])))
    style_to_label = {style: idx for idx, style in enumerate(unique_styles)}
    print("Style Labels:", style_to_label)

    with h5py.File('./100STYLE4pae.hdf5', 'w') as h5f:
        dset = None

        for filename, start, stop, style in files:
            for mirror in [False, True]:
                print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))

                bvh_data = bvh.load_zeroeggs(filename)
                parents = bvh_data['parents']
                bvh_data['positions'] = bvh_data['positions'][start:stop]
                bvh_data['rotations'] = bvh_data['rotations'][start:stop]

                positions = bvh_data['positions']
                # positions[..., 0] = -positions[..., 0]

                rotations = quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order'])
                # xform = quat.to_xform(rotations)
                # xform[..., 0, 1] = -xform[..., 0, 1]
                # xform[..., 0, 2] = -xform[..., 0, 2]
                # xform[..., 1, 0] = -xform[..., 1, 0]
                # xform[..., 2, 0] = -xform[..., 2, 0]
                # rotations = quat.from_xform(xform)
                rotations = quat.unroll(rotations)

                positions *= 0.01

                if mirror:
                    rotations, positions = animation_mirror(rotations, positions, bvh_data['names'],
                                                            bvh_data['parents'])
                    rotations = quat.unroll(rotations)

                """ Supersample """

                nframes = positions.shape[0]
                print('frames: ', nframes)
                nbones = positions.shape[1]

                # # Supersample data to 60 fps
                # original_times = np.linspace(0, nframes - 1, nframes)
                # sample_times = np.linspace(0, nframes - 1, int(0.9 * (nframes * 2 - 1)))  # Speed up data by 10%
                #
                # # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
                # positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape(
                #     [len(sample_times), nbones, 3])
                # rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape(
                #     [len(sample_times), nbones, 4])

                # Need to re-normalize after super-sampling
                rotations = quat.normalize(rotations)

                selected = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71]

                y = data_process(rotations, positions, parents, selected)

                if dset is None:
                    dset = h5f.create_dataset(
                        'data',
                        shape=(0, *y.shape[1:]),
                        maxshape=(None, *y.shape[1:]),
                        dtype=np.float32,
                        compression='gzip',
                        chunks=True
                    )
                dset.resize(dset.shape[0] + y.shape[0], axis=0)
                dset[-y.shape[0]:] = y

        print(dset.shape)
