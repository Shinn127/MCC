from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch

from DeepPhase.network import GNN


def infer_dimensions(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    return {
        "input_dim": state_dict["Xnorm"].shape[1],
        "output_dim": state_dict["Ynorm"].shape[1],
        "gating_hidden": state_dict["G1.weight"].shape[0],
        "gating_output": state_dict["G3.weight"].shape[0],
        "main_hidden": state_dict["E1.W"].shape[2],
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = script_dir / "model_epoch_150.pth"
    export_path = script_dir / "GNN_DP_100STY.onnx"
    motion_dim = 372
    phase_dim = 120
    dropout = 0.2

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    dimensions = infer_dimensions(state_dict)
    style_dim = dimensions["input_dim"] - motion_dim - phase_dim

    gating_indices = torch.arange(motion_dim, dimensions["input_dim"])
    main_indices = torch.cat([
        torch.arange(0, motion_dim),
        torch.arange(motion_dim + phase_dim, dimensions["input_dim"]),
    ])

    network = GNN(
        gating_indices=gating_indices,
        gating_input=len(gating_indices),
        gating_hidden=dimensions["gating_hidden"],
        gating_output=dimensions["gating_output"],
        main_indices=main_indices,
        main_input=len(main_indices),
        main_hidden=dimensions["main_hidden"],
        main_output=dimensions["output_dim"],
        dropout=dropout,
        input_norm=np.zeros((2, dimensions["input_dim"]), dtype=np.float32),
        output_norm=np.zeros((2, dimensions["output_dim"]), dtype=np.float32),
    )
    network.load_state_dict(state_dict)
    network.eval()

    x = torch.randn(1, motion_dim + phase_dim + style_dim)

    torch.onnx.export(
        network,
        (x,),
        str(export_path),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == '__main__':
    main()



