from __future__ import annotations

import numpy as np
import torch

from .models import MotionDiffusion


def infer_model_dimensions(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    output_dim, latent_dim = state_dict["out_process.weight"].shape
    return {
        "input_dim": state_dict["xf_proj.weight"].shape[1],
        "traj_dim": state_dict["traj_proj.weight"].shape[1],
        "pose_dim": state_dict["pose_proj.weight"].shape[1],
        "output_dim": output_dim,
        "latent_dim": latent_dim,
        "num_styles": state_dict["embed_style.action_embedding"].shape[0],
    }


def build_motion_diffusion(
    state_dict: dict[str, torch.Tensor],
    denoising_steps: int,
) -> tuple[MotionDiffusion, dict[str, int]]:
    dimensions = infer_model_dimensions(state_dict)
    model = MotionDiffusion(
        input_dim=dimensions["input_dim"],
        traj_dim=dimensions["traj_dim"],
        pose_dim=dimensions["pose_dim"],
        output_dim=dimensions["output_dim"],
        num_styles=dimensions["num_styles"],
        latent_dim=dimensions["latent_dim"],
        denoising_steps=denoising_steps,
        x_norm=np.zeros((2, dimensions["input_dim"]), dtype=np.float32),
        y_norm=np.zeros((2, dimensions["output_dim"]), dtype=np.float32),
    )
    model.load_state_dict(state_dict)
    return model, dimensions

