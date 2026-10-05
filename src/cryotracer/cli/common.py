import argparse
import math
from pathlib import Path

RF_DETR_CHECKPOINT = "Roboflow/rf-detr-base"
PREPROCESSING_CONFIG = {"downsample": 4, "normalize": True, "low_pass": 0.2}


def checkpoint_source(value: str) -> str | Path:
    """Preserve Hub URLs without downloading during argument parsing."""
    return value if value.startswith("hf://") else Path(value)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--batch-size", type=positive_int, default=8, help="Micrographs per batch."
    )
    runtime.add_argument(
        "--num-workers",
        type=nonnegative_int,
        default=4,
        help="Data-loader worker processes; 0 uses the main process.",
    )
    runtime.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for the workload.",
    )


def add_model_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    model = parser.add_argument_group("model initialization")
    requirement = (
        "Required without --checkpoint; forbidden with --checkpoint."
        if not required
        else "Required for RF-DETR initialization."
    )
    model.add_argument(
        "--num-queries",
        type=positive_int,
        required=required,
        help=f"Number of filament queries. {requirement}",
    )
    model.add_argument(
        "--num-points",
        type=positive_int,
        required=required,
        help=f"Points per polyline (at least 2). {requirement}",
    )


def add_training_arguments(parser: argparse.ArgumentParser, *, max_steps: int) -> None:
    training = parser.add_argument_group("training")
    training.add_argument(
        "--max-steps",
        type=positive_int,
        default=max_steps,
        help="Maximum optimizer steps.",
    )
    training.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1e-4,
        help="Initial learning rate.",
    )
    training.add_argument(
        "--val-every",
        type=positive_int,
        default=100,
        help="Validate every N optimizer steps.",
    )
    training.add_argument(
        "--accumulate-grad-batches",
        type=positive_int,
        default=1,
        help="Batches accumulated per optimizer step.",
    )
    training.add_argument(
        "--early-stopping-patience",
        type=nonnegative_int,
        default=10,
        help=(
            "Validation checks without improvement in val/loss before "
            "stopping (minimize); 0 disables early stopping."
        ),
    )
    training.add_argument(
        "--seed",
        type=nonnegative_int,
        default=42,
        help="Random seed, including synthetic generation or the 80/20 data split.",
    )
    parser.set_defaults(early_stopping_monitor="val/loss", early_stopping_mode="min")


def validate_initialization(args, parser):
    checkpoint = getattr(args, "checkpoint", None)
    loaded = checkpoint is not None
    options = {"--num-queries": args.num_queries, "--num-points": args.num_points}
    if loaded:
        supplied = [key for key, value in options.items() if value is not None]
        if supplied:
            parser.error(f"{' and '.join(supplied)} cannot be used with --checkpoint")
    else:
        missing = [key for key, value in options.items() if value is None]
        if missing:
            parser.error(f"{' and '.join(missing)} required without --checkpoint")
    if args.num_points is not None and args.num_points < 2:
        parser.error("--num-points must be at least 2")
    args.rf_detr_checkpoint = None if loaded else RF_DETR_CHECKPOINT
    args.preprocessing_config = None if loaded else PREPROCESSING_CONFIG.copy()
