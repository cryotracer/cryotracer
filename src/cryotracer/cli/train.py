import argparse
from pathlib import Path

from cryotracer.cli.common import (
    add_model_arguments,
    add_training_arguments,
    checkpoint_source,
    validate_initialization,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training input, checkpoint, model, and training options."""
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Directory of MRC files with sibling CBOX labels (searched recursively), "
            "or a Parquet metadata file."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=checkpoint_source,
        help=(
            "Local CryoTracer checkpoint or hf://owner/repo[@revision]/file.ckpt. "
            "Model and preprocessing settings "
            "are loaded unchanged; training controls remain editable."
        ),
    )
    add_model_arguments(parser, required=False)
    add_training_arguments(parser, max_steps=5000)
    parser.set_defaults(train_fraction=0.8)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Require fresh-model options or defer model configuration to a checkpoint."""
    validate_initialization(args, parser)


def run(args: argparse.Namespace) -> None:
    from cryotracer.train import run as train

    train(args)
