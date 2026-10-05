from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
from torch.utils.data import Dataset

from cryotracer.data.geometry import Polyline


@dataclass
class Micrograph:
    image_path: str
    cbox_path: str
    polylines: list[Polyline]
    shape: tuple[int, int]
    pixel_size_angstrom: float


def read_cbox(cbox_path: Path | str) -> dict[int, Polyline]:
    """Read ordered filament vertices while preserving their IDs."""
    blocks = starfile.read(cbox_path, always_dict=True)
    data = blocks.get("filament_vertices")
    if not isinstance(data, pd.DataFrame) or not {
        "CoordinateX",
        "CoordinateY",
        "filamentid",
    }.issubset(data.columns):
        raise ValueError(f"Missing filament vertices or columns in {cbox_path}")

    filaments = {}
    for filament_id, group in data.groupby("filamentid", sort=False, dropna=False):
        points = group[["CoordinateX", "CoordinateY"]].to_numpy(dtype=float)
        if len(points) < 2 or not np.isfinite(points).all():
            raise ValueError(f"Invalid filament vertices in {cbox_path}")
        filaments[filament_id] = [tuple(point) for point in points]
    return filaments


def parse_cbox(cbox_path: Path | str) -> list[Polyline]:
    return list(read_cbox(cbox_path).values())


def write_cbox(
    cbox_path: Path | str,
    polylines: list[Polyline],
    box_size: float = 400.0,
) -> None:
    """Write polylines as a filament-vertex CBOX file."""
    if not np.isfinite(box_size) or box_size <= 0:
        raise ValueError("box_size must be positive")

    rows = []
    for filament_id, polyline in enumerate(polylines, start=1):
        rows.extend(
            {
                "CoordinateX": float(x),
                "CoordinateY": float(y),
                "filamentid": filament_id,
                "Width": float(box_size),
                "Height": float(box_size),
            }
            for x, y in polyline
        )
    columns = ["CoordinateX", "CoordinateY", "filamentid", "Width", "Height"]
    table = pd.DataFrame(rows, columns=columns)
    path = Path(cbox_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"filament_vertices": table}, path, overwrite=True)


class CBoxDataset(Dataset):
    def __init__(
        self,
        micrographs: pd.DataFrame,
        transform: Callable,
        pixel_size_angstrom: float | None = None,
    ):
        if pixel_size_angstrom is not None and (
            not np.isfinite(pixel_size_angstrom) or pixel_size_angstrom <= 0
        ):
            raise ValueError("pixel_size_angstrom must be finite and positive")
        self.transform = transform
        self.pixel_size_angstrom = pixel_size_angstrom

        self.micrographs: list[Micrograph] = []

        for _, row in micrographs.reset_index(drop=True).iterrows():
            image_path = str(row["image_path"])
            cbox_path = str(row["cbox_path"])
            polylines = parse_cbox(cbox_path)

            if len(polylines) > transform.num_queries:
                raise ValueError(
                    f"{cbox_path}: {len(polylines)} filaments exceed num_queries={transform.num_queries}"
                )
            with mrcfile.mmap(image_path, mode="r") as mrc:
                if mrc.data.ndim != 2:
                    raise ValueError(f"Expected a 2D MRC image: {image_path}")
                height, width = mrc.data.shape
                image_pixel_size = float(mrc.voxel_size.x)
            effective_pixel_size = (
                float(pixel_size_angstrom)
                if pixel_size_angstrom is not None
                else image_pixel_size
            )

            if not np.isfinite(effective_pixel_size) or effective_pixel_size <= 0:
                raise ValueError(
                    f"MRC header must contain a finite positive pixel size: {image_path}"
                )

            self.micrographs.append(
                Micrograph(
                    image_path=image_path,
                    cbox_path=cbox_path,
                    polylines=polylines,
                    shape=(height, width),
                    pixel_size_angstrom=effective_pixel_size,
                )
            )

    def __len__(self) -> int:
        return len(self.micrographs)

    def __getitem__(self, idx: int) -> dict:
        micrograph = self.micrographs[idx]
        with mrcfile.mmap(micrograph.image_path, mode="r") as mrc:
            image = torch.from_numpy(mrc.data.copy()).float().unsqueeze(0)

        native_height, native_width = image.shape[-2:]
        out = self.transform(
            image,
            micrograph.polylines,
            pixel_size=micrograph.pixel_size_angstrom,
        )
        out["pixel_values"] = out.pop("image")
        valid = out["polylines"][:, :, 0].ne(-1).any(dim=1)
        out["class_labels"] = torch.full((len(valid),), -1, dtype=torch.long)
        out["class_labels"][valid] = 0
        out["image_id"] = torch.tensor(idx, dtype=torch.long)
        input_height, input_width = out["pixel_values"].shape[-2:]
        out["input_size"] = torch.tensor([input_width, input_height], dtype=torch.long)
        out["native_size"] = torch.tensor(
            [native_width, native_height], dtype=torch.long
        )
        return out
