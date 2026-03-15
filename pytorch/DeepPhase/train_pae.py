from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

import Library.Plotting as plot
import Library.Utility as utility
from DeepPhase.network import PAE
from Library.AdamWR import adamw, cyclic_scheduler
from Library.Training import choose_device, ensure_directory, load_hdf5_datasets, seed_everything


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="__array_wrap__ must accept context and return_scalar arguments",
)


def detach_to_cpu(value):
    return value.detach().cpu()


def main() -> None:
    window = 2.0
    fps = 60
    keys = 13
    joints = 24

    frames = int(window * fps) + 1
    input_channels = 3 * joints
    phase_channels = 5

    epochs = 30
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-4
    restart_period = 10
    restart_mult = 2

    plotting_interval = 250
    pca_sequence_count = 20

    seed_everything(1234)

    script_dir = Path(__file__).resolve().parent
    save_dir = ensure_directory(script_dir / "PAE_Training")
    data_path = script_dir / "100STYLE4pae.hdf5"

    data = load_hdf5_datasets(data_path, {"data": np.float32})["data"]
    print(data.shape)

    x = data.astype(np.float32)
    sample_count = x.shape[0]
    indices = torch.as_tensor(np.array([np.arange(i, i + frames) for i in range(sample_count - frames)]), dtype=torch.long)
    sample_count = len(indices)

    x = torch.as_tensor(x)
    print(x.shape)

    device = choose_device()
    print(device)

    network = PAE(
        input_channels=input_channels,
        embedding_channels=phase_channels,
        time_range=frames,
        key_range=keys,
        window=window,
    ).to(device)

    plt.ion()
    _, ax1 = plt.subplots(6, 1)
    _, ax2 = plt.subplots(phase_channels, 5)
    _, ax3 = plt.subplots(1, 2)
    _, ax4 = plt.subplots(2, 1)
    dist_amps = []
    dist_freqs = []
    loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, drawInterval=plotting_interval)

    print("Training Phases")
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

    shuffled_indices = np.arange(sample_count)
    ordered_indices = np.arange(sample_count)
    test_batch_size = min(batch_size, sample_count)

    for epoch in range(epochs):
        scheduler.step()
        np.random.shuffle(shuffled_indices)

        for start in range(0, sample_count, batch_size):
            train_indices = shuffled_indices[start:start + batch_size]
            if len(train_indices) == 0:
                continue

            train_batch = x[indices[train_indices]].to(device)
            train_batch = train_batch.permute(0, 3, 2, 1).reshape(-1, frames * joints * 3)

            y_pred, latent, signal, params = network(train_batch)

            loss = loss_function(y_pred, train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            amplitudes = detach_to_cpu(params[2]).squeeze().numpy()
            dist_amps.extend(amplitudes)
            dist_amps[:] = dist_amps[-10000:]

            frequencies = detach_to_cpu(params[1]).squeeze().numpy()
            dist_freqs.extend(frequencies)
            dist_freqs[:] = dist_freqs[-10000:]

            loss_history.Add((detach_to_cpu(loss).item(), "Reconstruction Loss"))

            if loss_history.Counter == 0:
                plot.Functions(
                    ax1[0],
                    detach_to_cpu(train_batch[0]).reshape(network.input_channels, frames),
                    -1.0,
                    1.0,
                    -5.0,
                    5.0,
                    title=f"Motion Curves {network.input_channels}x{frames}",
                    showAxes=False,
                )
                plot.Functions(
                    ax1[1],
                    detach_to_cpu(latent[0]),
                    -1.0,
                    1.0,
                    -2.0,
                    2.0,
                    title=f"Latent Convolutional Embedding {phase_channels}x{frames}",
                    showAxes=False,
                )
                plot.Circles(
                    ax1[2],
                    detach_to_cpu(params[0][0]).squeeze(),
                    detach_to_cpu(params[2][0]).squeeze(),
                    title=f"Learned Phase Timing {phase_channels}x2",
                    showAxes=False,
                )
                plot.Functions(
                    ax1[3],
                    detach_to_cpu(signal[0]),
                    -1.0,
                    1.0,
                    -2.0,
                    2.0,
                    title=f"Latent Parametrized Signal {phase_channels}x{frames}",
                    showAxes=False,
                )
                plot.Functions(
                    ax1[4],
                    detach_to_cpu(y_pred[0]).reshape(network.input_channels, frames),
                    -1.0,
                    1.0,
                    -5.0,
                    5.0,
                    title=f"Curve Reconstruction {network.input_channels}x{frames}",
                    showAxes=False,
                )
                plot.Function(
                    ax1[5],
                    [detach_to_cpu(train_batch[0]), detach_to_cpu(y_pred[0])],
                    -1.0,
                    1.0,
                    -5.0,
                    5.0,
                    colors=[(0, 0, 0), (0, 1, 1)],
                    title=f"Curve Reconstruction (Flattened) 1x{network.input_channels * frames}",
                    showAxes=False,
                )
                plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
                plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")

                with torch.no_grad():
                    test_start = random.randint(0, max(sample_count - test_batch_size, 0))
                    test_indices = ordered_indices[test_start:test_start + test_batch_size]
                    test_batch = x[indices[test_indices]].to(device)
                    test_batch = test_batch.permute(0, 3, 2, 1).reshape(-1, frames * joints * 3)
                    _, _, _, test_params = network(test_batch)

                    for channel_index in range(phase_channels):
                        phase = test_params[0][:, channel_index]
                        freq = test_params[1][:, channel_index]
                        amps = test_params[2][:, channel_index]
                        offs = test_params[3][:, channel_index]
                        plot.Phase1D(
                            ax2[channel_index, 0],
                            detach_to_cpu(phase),
                            detach_to_cpu(amps),
                            color=(0, 0, 0),
                            title=("1D Phase Values" if channel_index == 0 else None),
                            showAxes=False,
                        )
                        plot.Phase2D(
                            ax2[channel_index, 1],
                            detach_to_cpu(phase),
                            detach_to_cpu(amps),
                            title=("2D Phase Vectors" if channel_index == 0 else None),
                            showAxes=False,
                        )
                        plot.Functions(
                            ax2[channel_index, 2],
                            detach_to_cpu(freq).transpose(0, 1),
                            -1.0,
                            1.0,
                            0.0,
                            4.0,
                            title=("Frequencies" if channel_index == 0 else None),
                            showAxes=False,
                        )
                        plot.Functions(
                            ax2[channel_index, 3],
                            detach_to_cpu(amps).transpose(0, 1),
                            -1.0,
                            1.0,
                            0.0,
                            1.0,
                            title=("Amplitudes" if channel_index == 0 else None),
                            showAxes=False,
                        )
                        plot.Functions(
                            ax2[channel_index, 4],
                            detach_to_cpu(offs).transpose(0, 1),
                            -1.0,
                            1.0,
                            -1.0,
                            1.0,
                            title=("Offsets" if channel_index == 0 else None),
                            showAxes=False,
                        )

                    pca_indices = []
                    pca_batches = []
                    pivot = 0
                    for _ in range(pca_sequence_count):
                        test_start = random.randint(0, max(sample_count - test_batch_size, 0))
                        test_indices = ordered_indices[test_start:test_start + test_batch_size]
                        test_batch = x[indices[test_indices]].to(device)
                        test_batch = test_batch.permute(0, 3, 2, 1).reshape(-1, frames * joints * 3)
                        _, _, _, pca_params = network(test_batch)
                        amplitudes = detach_to_cpu(pca_params[2]).squeeze().numpy()
                        phase = detach_to_cpu(pca_params[0]).squeeze().numpy()
                        manifold_x = amplitudes * np.sin(2.0 * np.pi * phase)
                        manifold_y = amplitudes * np.cos(2.0 * np.pi * phase)
                        manifold = np.hstack((manifold_x, manifold_y))
                        pca_indices.append(pivot + np.arange(len(test_indices)))
                        pca_batches.append(manifold)
                        pivot += len(test_indices)

                plot.PCA2D(ax4[0], pca_indices, pca_batches, f"Phase Manifold ({pca_sequence_count} Random Sequences)")
                plt.gcf().canvas.draw_idle()

            plt.gcf().canvas.start_event_loop(1e-5)

        checkpoint_path = save_dir / f"{epoch + 1}_{phase_channels}Channels.pt"
        torch.save(network, checkpoint_path)
        print("Epoch", epoch + 1, loss_history.CumulativeValue())


if __name__ == '__main__':
    main()
