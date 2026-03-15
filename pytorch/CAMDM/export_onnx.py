from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch

from CAMDM.runtime import build_motion_diffusion

try:
    import onnx
except ImportError:
    onnx = None


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = script_dir / "camdm_4_epoch150.pt"
    export_path = script_dir / "camdm_4_epoch150.onnx"
    denoising_steps = 4

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model, dimensions = build_motion_diffusion(state_dict, denoising_steps=denoising_steps)
    model.eval()

    batch_size = 1
    noisy_target = torch.randn(batch_size, dimensions["output_dim"])
    x_batch = torch.randn(batch_size, dimensions["input_dim"])
    label_batch = torch.randint(0, dimensions["num_styles"], (batch_size,), dtype=torch.long)
    timesteps = torch.randint(0, denoising_steps, (batch_size,), dtype=torch.long)

    torch.onnx.export(
        model,
        (noisy_target, x_batch, label_batch, timesteps),
        str(export_path),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["x", "xf", "cond", "t"],
        output_names=["output"],
        dynamic_axes={
            "x": {0: "batch_size"},
            "xf": {0: "batch_size"},
            "cond": {0: "batch_size"},
            "t": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    if onnx is None:
        print(f"Exported ONNX model to {export_path} (validation skipped: onnx not installed)")
        return

    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX模型验证通过，导出成功：{export_path}")


if __name__ == '__main__':
    main()
