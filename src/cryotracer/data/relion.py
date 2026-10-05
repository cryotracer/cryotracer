import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

PARTICLE_COLUMNS = [
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnHelicalTubeID",
    "rlnAngleTiltPrior",
    "rlnAnglePsiPrior",
    "rlnHelicalTrackLengthAngst",
    "rlnAnglePsiFlipRatio",
]


def _resolved_micrograph_path(name: str, star_path: Path) -> Path:
    path = Path(name).expanduser()
    if path.is_absolute():
        return path.resolve()
    # RELION paths are relative to the project, often above CtfFind/jobXXX.
    for base in (star_path.parent, *star_path.parent.parents, Path.cwd()):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _pixel_size_column(optics: pd.DataFrame) -> str:
    for column in ("rlnMicrographPixelSize", "rlnImagePixelSize"):
        if column in optics:
            return column
    raise ValueError("RELION optics need rlnMicrographPixelSize or rlnImagePixelSize")


def read_relion_micrographs(star_path: Path, stems: list[str]) -> pd.DataFrame:
    """Match CBOX stems to micrographs and optics; CTF values are not required."""
    star_path = star_path.expanduser().resolve()
    blocks = starfile.read(star_path, always_dict=True)
    for name in ("optics", "micrographs"):
        if not isinstance(blocks.get(name), pd.DataFrame):
            raise ValueError(f"RELION STAR file is missing a data_{name} loop")  # noqa: TRY004
    optics = blocks["optics"]
    micrographs = blocks["micrographs"].copy()
    required = {
        "rlnMicrographName",
        "rlnOpticsGroup",
    }
    missing = sorted(required - set(micrographs.columns))
    if missing:
        raise ValueError("RELION micrographs are missing: " + ", ".join(missing))
    if "rlnOpticsGroup" not in optics:
        raise ValueError("RELION optics are missing rlnOpticsGroup")
    if optics["rlnOpticsGroup"].duplicated().any():
        raise ValueError("Duplicate RELION optics groups")
    micrographs.index = [
        _resolved_micrograph_path(str(name), star_path)
        for name in micrographs["rlnMicrographName"]
    ]
    if not micrographs.index.is_unique:
        raise ValueError("Duplicate RELION micrograph paths")
    wanted = set(stems)
    matches = {}
    for path in micrographs.index:
        if path.stem not in wanted:
            continue
        if path.stem in matches:
            raise ValueError(f"Ambiguous micrograph stem in {star_path}: {path.stem}")
        matches[path.stem] = path
    for stem in stems:
        if stem not in matches:
            raise ValueError(f"CBOX stem is not present in {star_path}: {stem}")
    paths = [matches[stem] for stem in stems]
    selected = micrographs.loc[paths].copy()
    pixels = optics.set_index("rlnOpticsGroup")[_pixel_size_column(optics)]
    selected["pixel_size"] = selected["rlnOpticsGroup"].map(pixels)
    if not (np.isfinite(selected["pixel_size"]) & (selected["pixel_size"] > 0)).all():
        raise ValueError(
            "Selected micrographs need an optics group with a positive pixel size"
        )
    return selected


def _cross(left, right):
    return left[0] * right[1] - left[1] * right[0]


def _crossing_distances(polylines, cumulative):
    distances = [[] for _ in polylines]
    for first, second in itertools.combinations(range(len(polylines)), 2):
        for i, a in enumerate(np.diff(polylines[first], axis=0)):
            for j, b in enumerate(np.diff(polylines[second], axis=0)):
                denominator = _cross(a, b)
                if abs(denominator) <= 1e-10:
                    continue
                offset = polylines[second][j] - polylines[first][i]
                t = _cross(offset, b) / denominator
                u = _cross(offset, a) / denominator
                if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
                    distances[first].append(
                        cumulative[first][i] + np.clip(t, 0, 1) * np.linalg.norm(a)
                    )
                    distances[second].append(
                        cumulative[second][j] + np.clip(u, 0, 1) * np.linalg.norm(b)
                    )
    return distances


def _interpolate(points, cumulative, distances):
    return np.column_stack(
        [np.interp(distances, cumulative, coordinate) for coordinate in points.T]
    )


def sample_relion_particles(
    polylines,
    *,
    tube_ids: list[int],
    image_shape: tuple[int, int],
    pixel_size: float,
    box_size: int,
    spacing_angstrom: float,
    crossing_exclusion_angstrom: float,
    bimodal_angular_priors: bool = False,
) -> pd.DataFrame:
    """Sample native coordinates without resetting distances after exclusions."""
    if not np.isfinite(spacing_angstrom) or spacing_angstrom <= 0:
        raise ValueError("Particle spacing must be finite and positive")
    if len(tube_ids) != len(polylines):
        raise ValueError("Each polyline needs a tube ID")
    if any(
        not isinstance(tube_id, (int, float, np.integer, np.floating))
        or not np.isfinite(tube_id)
        or tube_id < 1
        or int(tube_id) != tube_id
        for tube_id in tube_ids
    ) or len(set(tube_ids)) != len(tube_ids):
        raise ValueError("Tube IDs must be distinct positive integers")
    polylines = [np.asarray(line, dtype=float).reshape(-1, 2) for line in polylines]
    cumulative = [
        np.r_[0.0, np.linalg.norm(np.diff(line, axis=0), axis=1).cumsum()]
        for line in polylines
    ]
    crossings = (
        _crossing_distances(polylines, cumulative)
        if crossing_exclusion_angstrom > 0
        else [[] for _ in polylines]
    )
    spacing = spacing_angstrom / pixel_size
    radius = max(spacing / 2, 1.0)
    half_box = box_size / 2
    bounds = np.asarray(image_shape[::-1]) - half_box
    tables = []
    for index, (points, lengths) in enumerate(zip(polylines, cumulative)):
        if len(points) < 2 or lengths[-1] <= 1e-12:
            continue
        distances = np.arange(0.0, lengths[-1] + spacing * 1e-6, spacing)
        coordinates = _interpolate(points, lengths, distances)
        keep = ((coordinates >= half_box) & (coordinates <= bounds)).all(axis=1)
        for crossing in crossings[index]:
            keep &= (
                np.abs(distances - crossing) >= crossing_exclusion_angstrom / pixel_size
            )
        distances, coordinates = distances[keep], coordinates[keep]
        if len(distances) == 0:
            continue
        tangent = _interpolate(points, lengths, distances + radius) - _interpolate(
            points, lengths, distances - radius
        )
        if (np.linalg.norm(tangent, axis=1) <= 1e-12).any():
            raise ValueError("Cannot determine the local filament tangent")
        tables.append(
            pd.DataFrame(
                {
                    "rlnCoordinateX": coordinates[:, 0],
                    "rlnCoordinateY": coordinates[:, 1],
                    "rlnHelicalTubeID": int(tube_ids[index]),
                    "rlnAngleTiltPrior": 90.0,
                    "rlnAnglePsiPrior": -np.degrees(
                        np.arctan2(tangent[:, 1], tangent[:, 0])
                    ),
                    "rlnHelicalTrackLengthAngst": distances * pixel_size,
                    "rlnAnglePsiFlipRatio": 0.5 if bimodal_angular_priors else 0.0,
                }
            )
        )
    if not tables:
        return pd.DataFrame(columns=PARTICLE_COLUMNS)
    return pd.concat(tables, ignore_index=True)
