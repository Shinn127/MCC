from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
from tqdm import tqdm

import Library.Utility as utility
from Library.AdamWR import adamw, cyclic_scheduler
from Library.Training import choose_device, compute_normalization_stats, seed_everything
from MANN import MANN
from MANN.layouts import build_gating_indices


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "lafan1_mann.npz"
    checkpoint_path = script_dir / "mann_lafan1.pt"

    device = choose_device()
    print(device)
    seed_everything(23456)

    epochs = 150
    batch_size = 512
    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    data = np.load(data_path)
    data_x = data["X"].astype(np.float32)
    data_y = data["Y"].astype(np.float32)

    input_norm = compute_normalization_stats(data_x)
    output_norm = compute_normalization_stats(data_y)

    x = torch.as_tensor(data_x)
    y = torch.as_tensor(data_y)

    sample_count = len(x)
    print(sample_count)

    gating_index = build_gating_indices(x.shape[1])
    network = MANN(
        gating_indices=gating_index,
        gating_input=len(gating_index),
        gating_hidden=32,
        gating_output=8,
        main_input=x.shape[1],
        main_hidden=512,
        main_output=y.shape[1],
        input_norm=input_norm,
        output_norm=output_norm,
        dropout=0.2,
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

        with tqdm(total=sample_count, desc=f"Epoch {epoch + 1}/{epochs}", unit="sample") as pbar:
            for start in range(0, sample_count, batch_size):
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
                pbar.update(len(train_indices))
                pbar.set_postfix(loss=f"{loss.item():.6f}", avg=f"{running_loss / batch_count:.6f}")

        print(f"Epoch {epoch + 1}, Average Loss: {running_loss / max(batch_count, 1):.4f}")

    torch.save(network.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")


if __name__ == '__main__':
    main()
