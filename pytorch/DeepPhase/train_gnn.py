from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
from tqdm import tqdm

import Library.Utility as utility
from DeepPhase.network import GNN
from Library.AdamWR import adamw, cyclic_scheduler
from Library.Training import (
    choose_device,
    compute_normalization_stats,
    ensure_directory,
    load_hdf5_datasets,
    seed_everything,
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    save_dir = ensure_directory(script_dir / "GNN_Training")
    data_path = script_dir / "100STYLE4gnn.hdf5"

    device = choose_device()
    print(device)
    seed_everything(1234)

    datasets = load_hdf5_datasets(data_path, {"X": np.float32, "Y": np.float32})
    data_x = datasets["X"]
    data_y = datasets["Y"]

    input_norm = compute_normalization_stats(data_x)
    output_norm = compute_normalization_stats(data_y)

    x = torch.as_tensor(data_x)
    y = torch.as_tensor(data_y)

    epochs = 150
    batch_size = 512
    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    sample_count = len(x)
    motion_dim = 372
    phase_dim = 120
    style_dim = x.shape[1] - motion_dim - phase_dim

    gating_indices = torch.arange(motion_dim, x.shape[1])
    main_indices = torch.cat([
        torch.arange(0, motion_dim),
        torch.arange(motion_dim + phase_dim, x.shape[1]),
    ])

    network = GNN(
        gating_indices=gating_indices,
        gating_input=len(gating_indices),
        gating_hidden=64,
        gating_output=8,
        main_indices=main_indices,
        main_input=len(main_indices),
        main_hidden=1024,
        main_output=y.shape[1],
        dropout=0.2,
        input_norm=input_norm,
        output_norm=output_norm,
    ).to(device)

    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(
        optimizer=optimizer,
        batch_size=batch_size,
        epoch_size=sample_count,
        restart_period=restart_period,
        t_mult=restart_mult,
        policy="cosine",
        verbose=True,
    )
    loss_function = torch.nn.MSELoss()

    index_order = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(index_order)
        running_loss = 0.0
        batch_count = 0

        batch_iter = tqdm(
            range(0, sample_count, batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for start in batch_iter:
            train_indices = index_order[start:start + batch_size]
            if len(train_indices) == 0:
                continue

            x_batch = x[train_indices].to(device)
            y_batch = y[train_indices].to(device)

            y_pred = network(x_batch)
            loss = loss_function(utility.Normalize(y_pred, network.Ynorm), utility.Normalize(y_batch, network.Ynorm))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            running_loss += loss.item()
            batch_count += 1
            batch_iter.set_postfix(batch_loss=f"{loss.item():.6f}", avg_loss=f"{running_loss / batch_count:.6f}")

        if (epoch + 1) % 10 == 0:
            save_path = save_dir / f"model_epoch_{epoch + 1}.pth"
            torch.save(network.state_dict(), save_path)
            print(f"\nModel saved at {save_path}")

        print(f"Epoch {epoch + 1} Avg Loss: {running_loss / max(batch_count, 1):.6f}")


if __name__ == '__main__':
    main()
