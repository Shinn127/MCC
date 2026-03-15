from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
from tqdm import tqdm

from config import (
    CodebookMatchingModelConfig,
    CodebookMatchingTrainConfig,
    build_model,
    resolve_checkpoint_path,
    resolve_data_path,
    resolve_predictions_path,
)
from Library.Training import choose_device, compute_normalization_stats, load_hdf5_datasets, seed_everything


def load_model(model_path, input_norm, output_norm, device):
    model = build_model(input_norm, output_norm, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def extract_windows(data_x, data_y, num_windows=500, window_size=120):
    total_frames = data_x.shape[0]
    possible_starts = total_frames - window_size + 1
    if possible_starts < num_windows:
        raise ValueError("Not enough frames in the dataset to extract the requested windows")

    start_indices = np.random.choice(possible_starts, size=num_windows, replace=False)

    x_windows = []
    y_windows = []
    for start in start_indices:
        end = start + window_size
        x_windows.append(data_x[start:end])
        y_windows.append(data_y[start:end])

    return np.array(x_windows), np.array(y_windows)


def predict_windows(model, x_windows, device, batch_size=64):
    predicted_windows = []
    knn = torch.ones(1, device=device)

    with torch.inference_mode():
        for start in tqdm(range(0, len(x_windows), batch_size), desc="Predicting windows"):
            batch_windows = x_windows[start:start + batch_size]
            x_tensor = torch.as_tensor(batch_windows, dtype=torch.float32, device=device)
            flattened_x = x_tensor.reshape(-1, x_tensor.shape[-1])
            prediction, _ = model(flattened_x, knn=knn, t=None)
            prediction_np = prediction.cpu().numpy().reshape(x_tensor.shape[0], x_tensor.shape[1], -1)
            predicted_windows.append(prediction_np)

    return np.concatenate(predicted_windows, axis=0)


def main() -> None:
    device = choose_device()
    print(f"Using device: {device}")

    train_config = CodebookMatchingTrainConfig()
    model_config = CodebookMatchingModelConfig()
    data_path = resolve_data_path(__file__)
    checkpoint_path = resolve_checkpoint_path(__file__)
    predictions_path = resolve_predictions_path(__file__)

    seed_everything(train_config.seed)

    datasets = load_hdf5_datasets(data_path, {"X": np.float32, "Y": np.float32})
    data_x = datasets["X"]
    data_y = datasets["Y"]

    if data_x.shape[1] != model_config.input_dim or data_y.shape[1] != model_config.output_dim:
        raise ValueError(
            f"Unexpected dataset shape: X={data_x.shape[1]}, Y={data_y.shape[1]}, "
            f"expected X={model_config.input_dim}, Y={model_config.output_dim}"
        )

    input_norm = compute_normalization_stats(data_x)
    output_norm = compute_normalization_stats(data_y)
    model = load_model(checkpoint_path, input_norm, output_norm, device)

    x_windows, y_windows = extract_windows(
        data_x,
        data_y,
        num_windows=train_config.prediction_windows,
        window_size=train_config.prediction_window_size,
    )

    predicted_y_windows = predict_windows(
        model,
        x_windows,
        device,
        batch_size=train_config.prediction_batch_size,
    )
    print(predicted_y_windows.shape)
    print(y_windows.shape)

    np.savez(
        predictions_path,
        original_y=y_windows,
        predicted_y=predicted_y_windows,
        x_windows=x_windows,
    )
    print(f"Predictions saved to {predictions_path}")


if __name__ == '__main__':
    main()
