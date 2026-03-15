from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import h5py
import numpy as np
import pandas as pd
import scipy.signal as signal

import Utils.bvh as bvh
from config import build_lafan1_source_files, resolve_data_path, resolve_source_dir
from Utils import quat


np.set_printoptions(precision=6, suppress=True)
window = 60


def animation_mirror(lrot, lpos, names, parents):
    joints_mirror = np.array([(
        names.index('Left' + n[5:]) if n.startswith('Right') else (
            names.index('Right' + n[4:]) if n.startswith('Left') else names.index(n)
        )) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    grot, gpos = quat.fk(lrot, lpos, parents)
    gpos_mirror = mirror_pos * gpos[:, joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:, joints_mirror]))
    return quat.ik(grot_mirror, gpos_mirror, parents)


def data_process(rotations, positions, parents, style, selected):
    velocities = np.empty_like(positions)
    velocities[1:-1] = 0.5 * (positions[2:] - positions[1:-1]) * 60.0 + 0.5 * (positions[1:-1] - positions[:-2]) * 60.0
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])

    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
        0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.0
        + 0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.0
    )
    angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])

    global_rotations, global_positions, global_velocities, global_angular_velocities = quat.fk_vel(
        rotations, positions, velocities, angular_velocities, parents
    )

    direction = np.array([1, 0, 1]) * quat.mul_vec(global_rotations[:, 0:1], np.array([0.0, 0.0, 1.0]))
    direction = direction / np.sqrt(np.sum(np.square(direction), axis=-1))[..., np.newaxis]
    direction = signal.savgol_filter(direction, 61, 3, axis=0, mode='interp')
    direction = direction / np.sqrt(np.sum(np.square(direction), axis=-1))[..., np.newaxis]

    root_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), direction))
    root_velocity = np.array([1, 0, 1]) * quat.inv_mul_vec(root_rotation, global_velocities[:, 0:1])
    root_angular_velocity = np.degrees(quat.inv_mul_vec(root_rotation, global_angular_velocities[:, 0:1]))

    local_positions = global_positions.copy()
    local_positions[:, :, 0] = local_positions[:, :, 0] - local_positions[:, 0:1, 0]
    local_positions[:, :, 2] = local_positions[:, :, 2] - local_positions[:, 0:1, 2]
    local_positions = quat.inv_mul_vec(root_rotation, local_positions)

    local_rotations = quat.inv_mul(root_rotation, global_rotations)
    local_xy = quat.to_xform_xy(local_rotations)
    local_velocities = quat.inv_mul_vec(root_rotation, global_velocities)

    x_values, y_values = [], []
    for index in range(window, len(positions) - window - 1):
        root_position = np.array([1, 0, 1]) * quat.inv_mul_vec(
            root_rotation[index:index + 1],
            global_positions[index - window:index + window:10, 0:1] - global_positions[index:index + 1, 0:1],
        )
        root_direction = quat.inv_mul_vec(root_rotation[index:index + 1], direction[index - window:index + window:10])
        root_vel = quat.inv_mul_vec(root_rotation[index:index + 1], global_velocities[index - window:index + window:10, 0:1])
        speed = np.sqrt(np.sum(np.square(root_vel), axis=-1))

        x_values.append(np.hstack([
            root_position[:, :, 0].ravel(), root_position[:, :, 2].ravel(),
            root_direction[:, :, 0].ravel(), root_direction[:, :, 2].ravel(),
            root_vel[:, :, 0].ravel(), root_vel[:, :, 2].ravel(),
            speed.ravel(),
        ]))

        y_values.append(np.hstack([
            root_velocity[index - 1, :, 0].ravel(),
            root_angular_velocity[index - 1, :, 1].ravel(),
            root_velocity[index - 1, :, 2].ravel(),
            local_positions[index - 1, selected].ravel(),
            local_xy[index - 1, selected, :, 0].ravel(),
            local_xy[index - 1, selected, :, 1].ravel(),
            local_velocities[index - 1, selected].ravel(),
        ]))

    return np.array(x_values), np.array(y_values)


def load_frame_cuts(csv_path='./100STYLE/Frame_Cuts.csv', base_dir='./100STYLE', suffixes=None):
    if suffixes is None:
        suffixes = ['BR', 'BW', 'FR', 'FW', 'ID', 'SR', 'SW', 'TR1']

    data = pd.read_csv(csv_path)
    formatted_entries = []
    for _, row in data.iterrows():
        style_name = row['STYLE_NAME']
        style_dir = str(Path(base_dir) / style_name).replace('\\', '/')
        for suffix in suffixes:
            start_col = f"{suffix}_START"
            stop_col = f"{suffix}_STOP"
            if all(col in row for col in [start_col, stop_col]):
                start = row[start_col]
                stop = row[stop_col]
                if pd.notna(start) and pd.notna(stop):
                    bvh_file = f"{style_name}_{suffix}.bvh"
                    full_path = str(Path(style_dir) / bvh_file).replace('\\', '/')
                    formatted_entries.append((full_path, int(start), int(stop), style_name))
    return formatted_entries


def main() -> None:
    source_dir = resolve_source_dir(__file__)
    output_path = resolve_data_path(__file__)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = build_lafan1_source_files(source_dir)

    with h5py.File(output_path, 'w') as h5f:
        input_dset = None
        output_dset = None

        for filename, start, stop, style in files:
            for mirror in [False, True]:
                print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))

                bvh_data = bvh.load_zeroeggs(str(filename))
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

                print('frames: ', positions.shape[0])
                rotations = quat.normalize(rotations)
                selected = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 35, 36, 37, 38, 61, 62, 63, 64, 68, 69, 70, 71]
                x, y = data_process(rotations, positions, parents, style, selected)

                if input_dset is None:
                    input_dset = h5f.create_dataset(
                        'X',
                        shape=(0, *x.shape[1:]),
                        maxshape=(None, *x.shape[1:]),
                        dtype=np.float32,
                        compression='gzip',
                        chunks=True,
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
                        chunks=True,
                    )
                output_dset.resize(output_dset.shape[0] + y.shape[0], axis=0)
                output_dset[-y.shape[0]:] = y

            print(input_dset.shape)
            print(output_dset.shape)

    print(f"Saved dataset to {output_path}")


if __name__ == '__main__':
    main()
