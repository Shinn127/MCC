from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from CAMDM import DDPM, MotionDiffusion
from Library.AdamWR import adamw, cyclic_scheduler
from Library.Training import choose_device, compute_normalization_stats, load_hdf5_datasets, seed_everything


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / "100STYLE_re_label_camdm.hdf5"
    checkpoint_path = script_dir / "camdm_32.pt"

    device = choose_device()
    print(f"Using device: {device}")
    seed_everything(1234)

    epochs = 150
    batch_size = 512
    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2
    n_steps = 32

    datasets = load_hdf5_datasets(
        data_path,
        {"X": np.float32, "Y": np.float32, "Labels": np.int32},
    )
    data_x = datasets["X"]
    data_y = datasets["Y"]
    data_labels = datasets["Labels"]

    input_norm = compute_normalization_stats(data_x)
    output_norm = compute_normalization_stats(data_y)

    x = torch.as_tensor(data_x)
    y = torch.as_tensor(data_y)
    labels = torch.as_tensor(data_labels, dtype=torch.long).view(-1)

    sample_count = len(x)
    num_styles = int(data_labels.max()) + 1 if data_labels.size else 1
    print(sample_count)

    ddpm = DDPM(device=device, n_steps=n_steps)
    model = MotionDiffusion(
        input_dim=x.shape[1],
        traj_dim=84,
        pose_dim=288,
        output_dim=y.shape[1],
        num_styles=num_styles,
        latent_dim=256,
        denoising_steps=n_steps,
        x_norm=input_norm,
        y_norm=output_norm,
    ).to(device)

    optimizer = adamw.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(
        optimizer=optimizer,
        batch_size=batch_size,
        epoch_size=sample_count,
        restart_period=restart_period,
        t_mult=restart_mult,
        policy="cosine",
        verbose=True,
    )

    indices = np.arange(sample_count)
    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(indices)
        running_loss = 0.0
        batch_count = 0

        batch_iter = tqdm(
            range(0, sample_count, batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for start in batch_iter:
            train_indices = indices[start:start + batch_size]
            if len(train_indices) == 0:
                continue

            current_batch_size = len(train_indices)
            x_batch = x[train_indices].to(device)
            y_batch = y[train_indices].to(device)
            label_batch = labels[train_indices].to(device)

            timesteps = torch.randint(0, ddpm.n_steps, (current_batch_size,), device=device)
            noise = torch.randn_like(y_batch)
            noisy_target = ddpm.sample_forward(y_batch, timesteps, noise)

            y_pred = model(noisy_target, x_batch, label_batch, timesteps)
            loss = F.mse_loss(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            running_loss += loss.item()
            batch_count += 1
            batch_iter.set_postfix(loss=f"{loss.item():.6f}", avg=f"{running_loss / batch_count:.6f}")

        print(f"Epoch {epoch + 1} Avg Loss: {running_loss / max(batch_count, 1):.6f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")


if __name__ == '__main__':
    main()

