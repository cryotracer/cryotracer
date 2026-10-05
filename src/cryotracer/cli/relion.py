import argparse
from pathlib import Path

from cryotracer.cli.common import positive_float, positive_int


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CBOX input and helical sampling options."""
    parser.add_argument(
        "input", help="CBOX file, directory searched recursively, or quoted CBOX glob."
    )
    parser.add_argument(
        "--micrographs-star",
        type=Path,
        required=True,
        help="RELION micrographs STAR matching the CBOX stems; CTF values are optional.",
    )
    parser.add_argument(
        "--box-size",
        type=positive_int,
        required=True,
        help="RELION extraction box size in original image pixels (even integer).",
    )
    parser.add_argument(
        "--helical-rise-angstrom",
        type=positive_float,
        required=True,
        help="Helical rise in angstroms.",
    )
    parser.add_argument(
        "--helical-asym-units",
        type=positive_int,
        required=True,
        help="Asymmetric units between particle samples; spacing is this times rise.",
    )
    parser.add_argument(
        "--helical-bimodal-angular-priors",
        action="store_true",
        help="Use bimodal Psi priors (flip ratio 0.5).",
    )
    parser.add_argument(
        "--no-crossing-removal",
        action="store_true",
        help="Disable particle exclusion around filament crossings.",
    )
    parser.add_argument(
        "--crossing-exclusion-angstrom",
        type=positive_float,
        default=140.0,
        help="Track distance in angstroms excluded on each side of a crossing.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing coordinate outputs."
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.box_size % 2:
        parser.error("RELION box size must be an even integer")
    if not args.micrographs_star.expanduser().is_file():
        parser.error(f"Micrographs STAR file not found: {args.micrographs_star}")


def run(args: argparse.Namespace) -> None:
    from cryotracer.relion import run as convert

    convert(args)
