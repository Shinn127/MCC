from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch

from MANN import MANN
from MANN.layouts import build_gating_indices


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
    checkpoint_path = script_dir / "mann_100sty_re_10.pt"
    export_path = script_dir / "mann_100sty_re_10.onnx"

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    dimensions = infer_dimensions(state_dict)
    gating_index = build_gating_indices(dimensions["input_dim"])

    network = MANN(
        gating_indices=gating_index,
        gating_input=len(gating_index),
        gating_hidden=dimensions["gating_hidden"],
        gating_output=dimensions["gating_output"],
        main_input=dimensions["input_dim"],
        main_hidden=dimensions["main_hidden"],
        main_output=dimensions["output_dim"],
        input_norm=np.zeros((2, dimensions["input_dim"]), dtype=np.float32),
        output_norm=np.zeros((2, dimensions["output_dim"]), dtype=np.float32),
        dropout=0.2,
    )
    network.load_state_dict(state_dict)
    network.eval()

    x = torch.randn(1, dimensions["input_dim"])

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



