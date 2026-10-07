import argparse

from cryotracer.cli.common import checkpoint_source, positive_float, probability


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add prediction input, checkpoint, and output options."""
    parser.add_argument(
        "input",
        help=(
            "Quoted MRC glob, for example 'MotionCorr/job003/movies1/*.mrc'. "
            "Files ending in _PS.mrc are skipped."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=checkpoint_source,
        required=True,
        help=(
            "Local CryoTracer checkpoint or hf://owner/repo[@revision]/file.ckpt "
            "with model and preprocessing settings."
        ),
    )
    inference = parser.add_argument_group("prediction")
    inference.add_argument(
        "--box-size",
        type=positive_float,
        required=True,
        help="CBOX width in original image pixels.",
    )
    inference.add_argument(
        "--prediction-threshold",
        type=probability,
        help="Score threshold; omitted means use the checkpoint value.",
    )
    inference.add_argument(
        "--overwrite", action="store_true", help="Replace existing prediction outputs."
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Defer model and preprocessing configuration to the checkpoint."""
    args.rf_detr_checkpoint = None
    args.preprocessing_config = None


def run(args: argparse.Namespace) -> None:
    from cryotracer.predict import run as predict

    predict(args)
