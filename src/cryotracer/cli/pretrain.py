import argparse

from cryotracer.cli.common import (
    PREPROCESSING_CONFIG,
    add_model_arguments,
    add_training_arguments,
    positive_float,
    positive_int,
    probability,
    validate_initialization,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model, training, and synthetic generator options."""
    add_model_arguments(parser, required=True)
    add_training_arguments(parser, max_steps=20000)
    generator = parser.add_argument_group("synthetic generator")
    generator.add_argument(
        "--image-size",
        type=positive_int,
        default=4096,
        help="Native square image side in pixels; divisible by 4. Rendered at 1/4 size.",
    )
    generator.add_argument(
        "--train-samples",
        type=positive_int,
        default=8192,
        help="Synthetic training samples per epoch.",
    )
    generator.add_argument(
        "--val-samples",
        type=positive_int,
        default=512,
        help="Fixed synthetic validation samples.",
    )
    generator.add_argument(
        "--max-filaments",
        type=positive_int,
        default=30,
        help="Maximum filaments per synthetic image (1 to 30).",
    )
    generator.add_argument(
        "--empty-probability",
        type=probability,
        default=0.1,
        help="Probability of an empty synthetic image.",
    )
    generator.add_argument(
        "--pixel-size-angstrom",
        type=positive_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.65, 1.5),
        help="Native pixel size range in angstroms per pixel; equal bounds are allowed.",
    )
    generator.add_argument(
        "--filament-width-angstrom",
        type=positive_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(100.0, 200.0),
        help="Synthetic filament width range in angstroms; equal bounds are allowed.",
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate generator settings and configure fresh-model initialization."""
    validate_initialization(args, parser)
    if args.image_size % PREPROCESSING_CONFIG["downsample"]:
        parser.error("--image-size must be divisible by 4")
    if args.max_filaments > 30:
        parser.error("--max-filaments must be between 1 and 30")
    if args.num_queries is not None and args.max_filaments > args.num_queries:
        parser.error("--max-filaments must not exceed --num-queries")
    for name in ("pixel_size_angstrom", "filament_width_angstrom"):
        lower, upper = getattr(args, name)
        if lower > upper:
            parser.error(f"--{name.replace('_', '-')} requires MIN <= MAX")
    args.render_size = args.image_size // PREPROCESSING_CONFIG["downsample"]


def run(args: argparse.Namespace) -> None:
    from cryotracer.train import run as train

    train(args)
