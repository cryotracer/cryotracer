from collections.abc import Callable, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ParquetDataset(Dataset):
    """Read MRC paths and native (x, y) filament points relative to metadata.

    Required columns are ``id``, ``mrc_file``, ``width``, ``height``,
    ``pixel_size`` (angstroms), and ``filaments``. Each filament contains
    ``points`` as ordered coordinate pairs and ``class: filament``.
    An empty filament list represents a micrograph without labeled filaments.
    """

    def __init__(
        self,
        data_path: str | Path | Sequence[str | Path],
        transform: Callable,
    ):
        paths = [data_path] if isinstance(data_path, (str, Path)) else data_path
        self.data_paths = [Path(path).expanduser().resolve() for path in paths]
        if not self.data_paths:
            raise ValueError("At least one metadata path is required")
        self.transform = transform
        required = {"id", "mrc_file", "width", "height", "pixel_size", "filaments"}
        tables = []
        for path in self.data_paths:
            table = pd.read_parquet(path)
            missing = sorted(required - set(table.columns))
            if missing:
                raise ValueError(f"Missing metadata columns in {path}: {missing}")
            table["mrc_file"] = [
                str((path.parent / name).resolve()) for name in table["mrc_file"]
            ]
            tables.append(table)
        self.metadata = (
            pd.concat(tables, ignore_index=True)
            .sort_values("id")
            .reset_index(drop=True)
        )
        if self.metadata["id"].isna().any() or self.metadata["id"].duplicated().any():
            raise ValueError("Micrograph IDs must be non-null and unique")

        if self.metadata["mrc_file"].duplicated().any():
            raise ValueError("Micrograph paths must be unique")
        self.image_paths: list[Path] = []
        self.polylines: list[list[list[tuple[float, float]]]] = []
        for row in self.metadata.itertuples(index=False):
            if not np.isfinite(row.pixel_size) or row.pixel_size <= 0:
                raise ValueError(f"Invalid pixel_size for {row.id}")
            if any(
                not np.isfinite(size) or size <= 0 or int(size) != size
                for size in (row.width, row.height)
            ):
                raise ValueError(f"Invalid image dimensions for {row.id}")
            path = Path(row.mrc_file)
            if not path.is_file():
                raise FileNotFoundError(path)
            polylines = []
            for filament in row.filaments:
                if filament["class"] != "filament":
                    raise ValueError(f"Unsupported filament class for {row.id}")
                points = np.asarray(
                    filament["points"].tolist()
                    if isinstance(filament["points"], np.ndarray)
                    else filament["points"],
                    dtype=float,
                )
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or len(points) < 2
                    or not np.isfinite(points).all()
                ):
                    raise ValueError(f"Invalid filament points for {row.id}")
                polylines.append([(float(x), float(y)) for x, y in points])
            if len(polylines) > transform.num_queries:
                raise ValueError(
                    f"{row.id}: {len(polylines)} filaments exceed num_queries={transform.num_queries}"
                )
            self.image_paths.append(path)
            self.polylines.append(polylines)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        with mrcfile.mmap(self.image_paths[idx], mode="r") as mrc:
            if mrc.data.shape != (row.height, row.width):
                raise ValueError(f"MRC shape disagrees with metadata for {row.id}")
            image = torch.from_numpy(mrc.data.copy()).float().unsqueeze(0)
        out = self.transform(image, self.polylines[idx], pixel_size=row.pixel_size)
        out["pixel_values"] = out.pop("image")
        valid = out["polylines"][:, :, 0].ne(-1).any(dim=1)
        out["class_labels"] = torch.full((len(valid),), -1, dtype=torch.long)
        out["class_labels"][valid] = 0
        out["image_id"] = torch.tensor(idx, dtype=torch.long)
        height, width = out["pixel_values"].shape[-2:]
        out["input_size"] = torch.tensor([width, height], dtype=torch.long)
        out["native_size"] = torch.tensor([row.width, row.height], dtype=torch.long)
        return out
