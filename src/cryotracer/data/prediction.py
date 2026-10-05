import glob
import math
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import torch
from torch.utils.data import Dataset

from cryotracer.data.transforms import PolylineTransform


def expand_micrographs(pattern: str) -> list[Path]:
    paths = sorted(
        {
            Path(path).resolve()
            for path in glob.glob(str(Path(pattern).expanduser()), recursive=True)
        }
    )
    if not paths:
        raise ValueError(f"No micrographs match {pattern!r}")
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".mrc":
            raise ValueError(f"Expected an MRC file: {path}")
    return paths


@dataclass(frozen=True)
class Micrograph:
    path: Path
    shape: tuple[int, int]
    pixel_size: float


class PredictionDataset(Dataset):
    def __init__(self, paths: list[Path], transform: PolylineTransform):
        self.transform = transform
        self.micrographs = []
        for path in paths:
            with mrcfile.mmap(path, mode="r") as mrc:
                if mrc.data.ndim != 2:
                    raise ValueError(f"Expected a 2D MRC image: {path}")
                shape = tuple(mrc.data.shape)
                pixel_size = float(mrc.voxel_size.x)
            if not math.isfinite(pixel_size) or pixel_size <= 0:
                raise ValueError(
                    f"MRC header must contain a positive pixel size: {path}"
                )
            self.micrographs.append(Micrograph(path, shape, pixel_size))

    def __len__(self):
        return len(self.micrographs)

    def __getitem__(self, index):
        micrograph = self.micrographs[index]
        with mrcfile.mmap(micrograph.path, mode="r") as mrc:
            image = torch.from_numpy(mrc.data.copy()).float()[None]
        out = self.transform(image, [], pixel_size=micrograph.pixel_size)
        return {
            "pixel_values": out["image"],
            "image_id": torch.tensor(index),
            "native_size": torch.tensor(micrograph.shape[::-1]),
        }
