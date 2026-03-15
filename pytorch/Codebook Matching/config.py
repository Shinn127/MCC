from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import Modules
from Library.Project import resolve_script_path
from models import CodebookMatching


@dataclass(frozen=True)
class CodebookMatchingModelConfig:
    input_dim: int = 84
    output_dim: int = 291
    encoder_dim: int = 1024
    estimator_dim: int = 1024
    decoder_dim: int = 1024
    codebook_channels: int = 128
    codebook_dim: int = 8
    dropout: float = 0.2

    @property
    def codebook_size(self) -> int:
        return self.codebook_channels * self.codebook_dim


@dataclass(frozen=True)
class CodebookMatchingTrainConfig:
    seed: int = 1234
    epochs: int = 150
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    restart_period: int = 10
    restart_mult: int = 2
    prediction_windows: int = 500
    prediction_window_size: int = 120
    prediction_batch_size: int = 64


def build_model(input_norm, output_norm, device, model_config: CodebookMatchingModelConfig | None = None):
    config = model_config or CodebookMatchingModelConfig()
    model = CodebookMatching(
        encoder=Modules.LinearEncoder(
            config.input_dim + config.output_dim,
            config.encoder_dim,
            config.encoder_dim,
            config.codebook_size,
            config.dropout,
        ),
        estimator=Modules.LinearEncoder(
            config.input_dim,
            config.estimator_dim,
            config.estimator_dim,
            config.codebook_size,
            config.dropout,
        ),
        decoder=Modules.LinearEncoder(
            config.codebook_size,
            config.decoder_dim,
            config.decoder_dim,
            config.output_dim,
            config.dropout,
        ),
        xNorm=input_norm,
        yNorm=output_norm,
        codebook_channels=config.codebook_channels,
        codebook_dim=config.codebook_dim,
    )
    return model.to(device)


def resolve_data_path(current_file: str | Path) -> Path:
    return resolve_script_path(current_file, "lafan1_re_CM.hdf5", "CODEBOOK_MATCHING_DATA_PATH")


def resolve_checkpoint_path(current_file: str | Path) -> Path:
    return resolve_script_path(
        current_file,
        "models/CM_lafan1.pt",
        "CODEBOOK_MATCHING_CHECKPOINT_PATH",
        create_parent=True,
    )


def resolve_predictions_path(current_file: str | Path) -> Path:
    return resolve_script_path(
        current_file,
        "predictions/predictions_results_lafna1.npz",
        "CODEBOOK_MATCHING_PREDICTIONS_PATH",
        create_parent=True,
    )


def resolve_source_dir(current_file: str | Path) -> Path:
    return resolve_script_path(current_file, "lafan1_re", "CODEBOOK_MATCHING_SOURCE_DIR")


def build_lafan1_source_files(source_dir: Path) -> list[tuple[Path, int, int, str]]:
    file_specs = [
        ("walk1_subject1.bvh", 189, 15620, "walk"),
        ("walk1_subject2.bvh", 160, 15504, "walk"),
        ("walk1_subject5.bvh", 174, 15489, "walk"),
        ("walk2_subject1.bvh", 225, 14224, "walk"),
        ("walk2_subject3.bvh", 198, 14197, "walk"),
        ("walk2_subject4.bvh", 238, 14184, "walk"),
        ("walk3_subject1.bvh", 151, 14684, "walk"),
        ("walk3_subject2.bvh", 137, 14726, "walk"),
        ("walk3_subject3.bvh", 164, 14698, "walk"),
        ("walk3_subject4.bvh", 178, 14657, "walk"),
        ("walk3_subject5.bvh", 178, 14602, "walk"),
        ("walk4_subject1.bvh", 219, 9725, "walk"),
    ]
    return [(source_dir / filename, start, stop, style) for filename, start, stop, style in file_specs]

