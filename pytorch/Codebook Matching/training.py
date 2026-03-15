from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import Library.Utility as utility
from config import (
    CodebookMatchingModelConfig,
    CodebookMatchingTrainConfig,
    build_model,
    resolve_checkpoint_path,
    resolve_data_path,
)
from Library.AdamWR import adamw, cyclic_scheduler
from Library.Training import choose_device, compute_normalization_stats, load_hdf5_datasets, seed_everything


def main() -> None:
    device = choose_device()
    print(f"Using device: {device}")

    train_config = CodebookMatchingTrainConfig()
    model_config = CodebookMatchingModelConfig()
    data_path = resolve_data_path(__file__)
    checkpoint_path = resolve_checkpoint_path(__file__)

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

    x = torch.as_tensor(data_x)
    y = torch.as_tensor(data_y)
    sample_count = len(x)
    print(sample_count)

    model = build_model(input_norm, output_norm, device, model_config=model_config)

    optimizer = adamw.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(
        optimizer=optimizer,
        batch_size=train_config.batch_size,
        epoch_size=sample_count,
        restart_period=train_config.restart_period,
        t_mult=train_config.restart_mult,
        policy="cosine",
        verbose=True,
    )
    loss_function = nn.MSELoss()
    knn = torch.ones(1, device=device)

    index_order = np.arange(sample_count)
    for epoch in range(train_config.epochs):
        scheduler.step()
        np.random.shuffle(index_order)
        running_loss = 0.0
        running_mse = 0.0
        batch_count = 0

        with tqdm(total=sample_count, desc=f"Epoch {epoch + 1}/{train_config.epochs}", unit="sample") as pbar:
            for start in range(0, sample_count, train_config.batch_size):
                train_indices = index_order[start:start + train_config.batch_size]
                if len(train_indices) == 0:
                    continue

                x_batch = x[train_indices].to(device)
                y_batch = y[train_indices].to(device)

                prediction, _, _, target, _, _, estimate = model(x_batch, knn=knn, t=y_batch)

                mse_loss = loss_function(utility.Normalize(y_batch, model.YNorm), utility.Normalize(prediction, model.YNorm))
                matching_loss = loss_function(target, estimate)
                loss = mse_loss + matching_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.batch_step()

                running_loss += loss.item()
                running_mse += mse_loss.item()
                batch_count += 1
                pbar.update(len(train_indices))
                pbar.set_postfix(loss=f"{loss.item():.6f}", avg=f"{running_loss / batch_count:.6f}")

        print(
            f"Epoch {epoch + 1}, Avg Loss: {running_loss / max(batch_count, 1):.4f}, "
            f"Avg MSE: {running_mse / max(batch_count, 1):.4f}"
        )

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")


if __name__ == '__main__':
    main()
