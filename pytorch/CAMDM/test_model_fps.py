from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import time

import numpy as np
import torch
from tqdm import tqdm

from CAMDM import DDPM
from CAMDM.runtime import build_motion_diffusion


def test_model_fps(
    model_path: str | Path,
    device: str = "cuda:0",
    num_tests: int = 1000,
    batch_size: int = 1,
    denoising_steps: int = 4,
    collect_output: bool = False,
):
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        runtime_device = torch.device("cpu")

    state_dict = torch.load(model_path, map_location=runtime_device)
    model, dimensions = build_motion_diffusion(state_dict, denoising_steps=denoising_steps)
    model = model.to(runtime_device)
    model.eval()

    ddpm = DDPM(device=runtime_device, n_steps=denoising_steps)
    dummy_x = torch.randn(batch_size, dimensions["input_dim"], dtype=torch.float32, device=runtime_device)
    dummy_y = torch.randn(batch_size, dimensions["output_dim"], dtype=torch.float32, device=runtime_device)
    dummy_labels = torch.randint(
        0,
        dimensions["num_styles"],
        (batch_size,),
        dtype=torch.long,
        device=runtime_device,
    )

    print("Warming up...")
    with torch.inference_mode():
        for _ in range(10):
            timesteps = torch.randint(0, ddpm.n_steps, (batch_size,), dtype=torch.long, device=runtime_device)
            noise = torch.randn_like(dummy_y)
            noisy_target = ddpm.sample_forward(dummy_y, timesteps, noise)
            _ = model(noisy_target, dummy_x, dummy_labels, timesteps)

    print(f"Testing inference speed with {num_tests} iterations...")
    latencies = []
    sample_output = None

    with torch.inference_mode():
        for _ in tqdm(range(num_tests)):
            timesteps = torch.randint(0, ddpm.n_steps, (batch_size,), dtype=torch.long, device=runtime_device)
            noise = torch.randn_like(dummy_y)
            noisy_target = ddpm.sample_forward(dummy_y, timesteps, noise)

            if runtime_device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            output = model(noisy_target, dummy_x, dummy_labels, timesteps)
            if runtime_device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start_time)

            if collect_output and sample_output is None:
                sample_output = output.cpu().numpy()

    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    fps = 1 / avg_latency

    print("\nPerformance Results:")
    print(f"- Average latency: {avg_latency * 1000:.2f} ms")
    print(f"- Frames per second: {fps:.2f} FPS")
    print(f"- Batch size: {batch_size}")
    print(f"- 95th percentile latency: {np.percentile(latencies, 95) * 1000:.2f} ms")

    return fps, sample_output


if __name__ == '__main__':
    checkpoint_path = Path(__file__).resolve().parent / "camdm_4.pt"
    fps, sample_output = test_model_fps(checkpoint_path, collect_output=True)
    if sample_output is not None:
        print(f"\nOutput shape: {sample_output.shape}")
