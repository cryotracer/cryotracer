import argparse
from importlib.metadata import version
from pathlib import Path

from cryotracer.cli import predict, pretrain, relion, train
from cryotracer.cli.common import add_runtime_arguments


def build_parser() -> argparse.ArgumentParser:
    """Build the lightweight CLI without importing processing dependencies."""
    parser = argparse.ArgumentParser(
        prog="cryotracer",
        description="Pretrain, train, predict filaments, and convert picks for RELION.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('cryotracer')}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = (
        (
            "pretrain",
            "Pretrain on synthetic micrographs from RF-DETR weights.",
            pretrain,
        ),
        (
            "train",
            "Train on MRC/CBOX pairs or Parquet metadata with an 80/20 split.",
            train,
        ),
        (
            "predict",
            "Predict from a local or Hugging Face checkpoint and export CBOX files.",
            predict,
        ),
        (
            "relion",
            "Convert CBOX filaments to sampled helical coordinates for RELION extraction.",
            relion,
        ),
    )
    for command, description, module in descriptions:
        subparser = commands.add_parser(
            command,
            help=description,
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False,
        )
        subparser.set_defaults(
            handler=module.run, _validate=module.validate_args, _parser=subparser
        )
        subparser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Directory for command outputs.",
        )
        if module is not relion:
            add_runtime_arguments(subparser)
        module.add_arguments(subparser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the API without opening data or checkpoint files."""
    args = build_parser().parse_args(argv)
    parser = args._parser
    validate = args._validate
    del args._parser
    del args._validate
    validate(args, parser)
    return args


def main(argv: list[str] | None = None) -> None:
    """Validate arguments and import the selected backend lazily."""
    args = parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, OSError) as error:
        raise SystemExit(f"cryotracer {args.command}: {error}") from None
