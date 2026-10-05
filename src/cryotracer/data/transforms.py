import math

import torch
import torchvision.transforms.v2.functional as TF
from torch import nn

from cryotracer.data.geometry import Polyline, clip_polyline, sample_polyline


class PolylineTransform(nn.Module):
    def __init__(
        self,
        num_queries: int,
        num_points: int,
        downsample: int | None = None,
        horizontal_flip_p: float = 0.0,
        vertical_flip_p: float = 0.0,
        rotation_degrees: float = 0.0,
        normalize: bool = True,
        low_pass: float | None = 0.2,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_points = num_points

        if downsample is not None and downsample <= 0:
            raise ValueError("downsample must be positive")
        self.downsample = downsample
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.rotation_degrees = rotation_degrees
        self.normalize = normalize
        self.low_pass = low_pass

    def forward(
        self,
        image: torch.Tensor,
        polylines: list[Polyline],
        pixel_size: float = 1,
    ) -> dict[str, torch.Tensor]:
        if len(polylines) > self.num_queries:
            raise ValueError(
                f"Sample has {len(polylines)} filaments but num_queries={self.num_queries}"
            )
        if not torch.isfinite(image).all():
            raise ValueError("Micrograph contains non-finite pixels")
        polylines = [list(polyline) for polyline in polylines]
        image, polylines = self.horizontal_flip(image, polylines)
        image, polylines = self.vertical_flip(image, polylines)
        image, polylines = self.resize(image, polylines)
        image, polylines = self.rotate(image, polylines)

        downsample_factor = self.downsample or 1
        effective_pixel_size = pixel_size * downsample_factor
        image = self.preprocess_image(image, pixel_size=effective_pixel_size)
        packed_polylines = self.pack_polylines(polylines, image.shape[-2:])

        return {
            "image": image,
            "polylines": packed_polylines,
            "pixel_size_angstrom": torch.tensor(
                effective_pixel_size, dtype=torch.float32
            ),
        }

    def horizontal_flip(
        self, image: torch.Tensor, polylines: list[Polyline]
    ) -> tuple[torch.Tensor, list[Polyline]]:
        if (
            self.horizontal_flip_p <= 0
            or torch.rand(1).item() >= self.horizontal_flip_p
        ):
            return image, polylines

        width = image.shape[-1]
        polylines = [
            [(width - 1 - x, y) for x, y in polyline] for polyline in polylines
        ]
        return TF.horizontal_flip(image), polylines

    def vertical_flip(
        self, image: torch.Tensor, polylines: list[Polyline]
    ) -> tuple[torch.Tensor, list[Polyline]]:
        if self.vertical_flip_p <= 0 or torch.rand(1).item() >= self.vertical_flip_p:
            return image, polylines

        height = image.shape[-2]
        polylines = [
            [(x, height - 1 - y) for x, y in polyline] for polyline in polylines
        ]
        return TF.vertical_flip(image), polylines

    def rotate(
        self, image: torch.Tensor, polylines: list[Polyline]
    ) -> tuple[torch.Tensor, list[Polyline]]:
        if self.rotation_degrees <= 0:
            return image, polylines

        angle = (
            torch.empty(1)
            .uniform_(-self.rotation_degrees, self.rotation_degrees)
            .item()
        )
        height, width = image.shape[-2:]
        polylines = self.rotate_polylines(polylines, angle, height=height, width=width)
        return TF.rotate(image, angle), polylines

    def rotate_polylines(
        self,
        polylines: list[Polyline],
        angle: float,
        height: int,
        width: int,
    ) -> list[Polyline]:
        center_x = width / 2
        center_y = height / 2
        radians = math.radians(-angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)

        rotated = []
        for polyline in polylines:
            points = []
            for x, y in polyline:
                x_centered = x - center_x
                y_centered = y - center_y
                points.append(
                    (
                        x_centered * cos_a - y_centered * sin_a + center_x,
                        x_centered * sin_a + y_centered * cos_a + center_y,
                    )
                )

            clipped = clip_polyline(points, (0, 0, width, height))
            if clipped is not None:
                rotated.append(clipped)

        return rotated

    def resize(
        self, image: torch.Tensor, polylines: list[Polyline]
    ) -> tuple[torch.Tensor, list[Polyline]]:
        if self.downsample is None:
            return image, polylines

        height, width = image.shape[-2:]
        size = [height // self.downsample, width // self.downsample]
        if min(size) < 1:
            raise ValueError("Micrograph is smaller than the downsampling factor")
        image = TF.resize(
            image,
            size,
            TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        polylines = [
            [(x * size[1] / width, y * size[0] / height) for x, y in polyline]
            for polyline in polylines
        ]
        return image, polylines

    def preprocess_image(self, image: torch.Tensor, pixel_size: float) -> torch.Tensor:
        valid_mask = image != 0

        if self.low_pass is not None:
            image = self.low_pass_filter(
                image, pixel_size=pixel_size, cutoff_abs=self.low_pass
            )

        if not self.normalize:
            image[~valid_mask] = 0
            return image

        valid_values = image[valid_mask]
        if valid_values.numel() == 0:
            return torch.full_like(image, -5)
        mean = valid_values.mean()
        std = valid_values.std(correction=0).clamp(min=1e-2)
        image = (image - mean) / std
        image = image.clamp(-5, 5)
        image[~valid_mask] = -5
        return image

    def low_pass_filter(
        self, image: torch.Tensor, pixel_size: float, cutoff_abs: float
    ) -> torch.Tensor:
        _, height, width = image.shape
        device = image.device

        fft = torch.fft.fft2(image)
        fy = torch.fft.fftfreq(height, d=pixel_size).to(device)
        fx = torch.fft.fftfreq(width, d=pixel_size).to(device)
        fy, fx = torch.meshgrid(fy, fx, indexing="ij")
        mask = torch.sqrt(fx**2 + fy**2) <= cutoff_abs

        return torch.fft.ifft2(fft * mask).real

    def pack_polylines(
        self, polylines: list[Polyline], image_shape: tuple[int, int]
    ) -> torch.Tensor:
        out = torch.full(
            (self.num_queries, self.num_points, 2), -1.0, dtype=torch.float32
        )
        height, width = image_shape
        scale = torch.tensor([width, height], dtype=torch.float32)

        for i, polyline in enumerate(polylines):
            out[i] = (sample_polyline(polyline, self.num_points) / scale).clamp(0, 1)

        return out
