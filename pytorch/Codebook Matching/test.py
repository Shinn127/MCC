from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
from tqdm import tqdm

from config import resolve_predictions_path
from Utils import quat


def fk_from_root(x, nbones=22, dt=1.0 / 60.0):
    sample_count, sequence_length = x.shape[0], x.shape[1]

    root_motion = x[:, :, :3] * dt
    local_motion = x[:, :, 3:].reshape(sample_count, sequence_length, 4, nbones, 3)

    translational_offset = root_motion.copy()
    translational_offset[:, :, 1] = 0.0

    rotational_offset = root_motion[:, :, 1].copy()
    rotational_offset = quat.from_angle_axis(rotational_offset, np.array([0, 1, 0]))

    root_position = np.zeros((sample_count, 3))
    root_rotation = np.array([1.0, 0.0, 0.0, 0.0]).reshape(1, -1).repeat(sample_count, axis=0)
    global_positions = local_motion[:, :, 0].copy()
    local_rotations_quat = quat.from_xform_xy(local_motion[:, :, 1:3].transpose(0, 1, 3, 4, 2))

    root_positions = np.zeros((sample_count, sequence_length, 3))
    root_rotations = np.zeros((sample_count, sequence_length, 4))

    for index in range(sequence_length):
        root_position += translational_offset[:, index]
        root_positions[:, index] = root_position

        root_rotation = quat.mul(rotational_offset[:, index], root_rotation)
        root_rotations[:, index] = root_rotation

    global_positions += np.expand_dims(root_positions, axis=2)
    global_rotations = quat.mul(np.expand_dims(root_rotations, axis=2), local_rotations_quat)
    global_velocities = quat.mul_vec(np.expand_dims(root_rotations, axis=2), local_motion[:, :, 3].copy())

    return global_positions, global_rotations, global_velocities


def bootstrap_ci(data, n_bootstrap=5000, ci=95):
    boot_means = []
    sample_count = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=sample_count, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, ci + (100 - ci) / 2)
    return lower, upper


def calculate_metrics(original_data, predicted_data):
    ape_samples = []
    for index in tqdm(range(original_data.shape[0]), desc="Calculating APE"):
        ape = np.mean(np.abs(original_data[index] * 100 - predicted_data[index] * 100))
        ape_samples.append(ape)

    ave_samples = []
    for index in tqdm(range(original_data.shape[0]), desc="Calculating AVE"):
        true_var = np.var(original_data[index], axis=0)
        pred_var = np.var(predicted_data[index], axis=0)
        ave = np.mean(np.abs(true_var * 100 - pred_var * 100))
        ave_samples.append(ave)

    return {
        'APE': (np.mean(ape_samples), bootstrap_ci(ape_samples)),
        'AVE': (np.mean(ave_samples), bootstrap_ci(ave_samples)),
    }


def print_results(results, model_name):
    ape_mean, (ape_low, ape_high) = results['APE']
    ape_half_width = (ape_high - ape_low) / 2
    ave_mean, (ave_low, ave_high) = results['AVE']
    ave_half_width = (ave_high - ave_low) / 2

    if model_name.lower() == 'codebook':
        print(f"\n{'Model':<10} | {'APE (cm)':<15} | {'AVE (cm)':<15}")
        print('-' * 40)

    print(f"{model_name:<10} | {ape_mean:.3f}±{ape_half_width:.3f}   | {ave_mean:.3f}±{ave_half_width:.3f}")


def main() -> None:
    predictions_path = resolve_predictions_path(__file__)
    prediction_bundle = np.load(predictions_path)
    original_data = prediction_bundle['original_y']
    predicted_data = prediction_bundle['predicted_y']

    original_gp, _, _ = fk_from_root(original_data, 24)
    predicted_gp, _, _ = fk_from_root(predicted_data, 24)
    results = calculate_metrics(original_gp, predicted_gp)
    print_results(results, 'Codebook')


if __name__ == '__main__':
    main()
