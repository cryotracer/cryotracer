import glob
import math
from pathlib import Path

import mrcfile
import pandas as pd
import starfile
from tqdm.auto import tqdm

from cryotracer.data.cbox import read_cbox
from cryotracer.data.relion import read_relion_micrographs, sample_relion_particles


def expand_cboxes(pattern: str) -> list[Path]:
    source = Path(pattern).expanduser()
    matches = (
        source.rglob("*.cbox")
        if source.is_dir()
        else glob.glob(str(source), recursive=True)
    )
    paths = sorted({Path(path).resolve() for path in matches})
    if not paths:
        raise ValueError(f"No CBOX files match {pattern!r}")
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".cbox":
            raise ValueError(f"Expected a CBOX file: {path}")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Input CBOX filenames must have unique stems")
    return paths


def run(args):
    cbox_paths = expand_cboxes(args.input)
    metadata = read_relion_micrographs(
        args.micrographs_star, [path.stem for path in cbox_paths]
    )
    output_dir = args.output_dir.expanduser().resolve()
    coordinate_paths = [
        output_dir / "coords" / f"{path.stem}_coords.star" for path in cbox_paths
    ]
    list_path = output_dir / "coordinates.star"
    inputs = {
        *cbox_paths,
        *metadata.index,
        args.micrographs_star.expanduser().resolve(),
    }
    for path in [*coordinate_paths, list_path]:
        if path.resolve() in inputs:
            raise ValueError(f"Output would overwrite an input: {path}")
        if path.is_dir():
            raise IsADirectoryError(path)
        if (path.exists() or path.is_symlink()) and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite")

    particle_count = tube_count = 0
    for cbox_path, coordinate_path, micrograph_path in tqdm(
        zip(cbox_paths, coordinate_paths, metadata.index),
        total=len(cbox_paths),
        desc="Converting",
        unit="micrograph",
    ):
        filaments = read_cbox(cbox_path)
        with mrcfile.mmap(micrograph_path, mode="r") as mrc:
            if mrc.data.ndim != 2:
                raise ValueError(f"Expected a 2D MRC image: {micrograph_path}")
            image_shape = tuple(mrc.data.shape)
            pixel_size = float(mrc.voxel_size.x)
        if not math.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(
                f"MRC header must contain a positive pixel size: {micrograph_path}"
            )
        if not math.isclose(
            pixel_size,
            metadata.loc[micrograph_path, "pixel_size"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise ValueError(f"MRC and RELION pixel sizes differ: {micrograph_path}")
        table = sample_relion_particles(
            list(filaments.values()),
            tube_ids=list(filaments),
            image_shape=image_shape,
            pixel_size=pixel_size,
            box_size=args.box_size,
            spacing_angstrom=args.helical_rise_angstrom * args.helical_asym_units,
            crossing_exclusion_angstrom=(
                0 if args.no_crossing_removal else args.crossing_exclusion_angstrom
            ),
            bimodal_angular_priors=args.helical_bimodal_angular_priors,
        )
        coordinate_path.parent.mkdir(parents=True, exist_ok=True)
        starfile.write(table, coordinate_path, overwrite=args.overwrite)
        particle_count += len(table)
        tube_count += table["rlnHelicalTubeID"].nunique()

    coordinate_list = pd.DataFrame(
        {
            "rlnMicrographName": metadata["rlnMicrographName"].to_list(),
            "rlnMicrographCoordinates": [str(path) for path in coordinate_paths],
        }
    )
    starfile.write(
        {"coordinate_files": coordinate_list}, list_path, overwrite=args.overwrite
    )
    print(
        f"Saved {particle_count} coordinates from {tube_count} filaments "
        f"in {len(coordinate_paths)} STAR files; RELION coordinate list: {list_path}"
    )
