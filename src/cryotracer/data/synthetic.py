from collections.abc import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from cryotracer.data.geometry import Polyline, clip_polyline


class SyntheticFilamentDataset(Dataset):
    """Deterministic-per-epoch full-frame synthetic filament micrographs.

    A sample is a pure function of ``(seed, epoch, index)``. This keeps labels,
    noise, and augmentations identical between matched ablation arms while
    generating a fresh training sample for every epoch. Plain integer indices
    use epoch zero so validation samples remain fixed.
    """

    def __init__(
        self,
        transform: Callable,
        image_size: int = 4096,
        num_samples: int = 128,
        seed: int = 42,
        max_filaments: int = 30,
        empty_probability: float = 0.1,
        pixel_size_angstrom: float | tuple[float, float] = (0.65, 1.5),
        render_size: int | None = None,
        filament_width_angstrom: float | tuple[float, float] = (100.0, 200.0),
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 0 < max_filaments <= 30:
            raise ValueError("max_filaments must be between 1 and 30")
        if not 0 <= empty_probability <= 1:
            raise ValueError("empty_probability must be between 0 and 1")
        render_size = render_size or image_size
        if render_size <= 0 or image_size % render_size != 0:
            raise ValueError("render_size must be a positive divisor of image_size")

        if max_filaments > transform.num_queries:
            raise ValueError("max_filaments must not exceed num_queries")
        self.transform = transform
        self.image_size = int(image_size)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.max_filaments = int(max_filaments)
        self.empty_probability = float(empty_probability)
        self.pixel_size_angstrom = (
            (pixel_size_angstrom, pixel_size_angstrom)
            if isinstance(pixel_size_angstrom, (int, float))
            else pixel_size_angstrom
        )
        self.render_size = int(render_size)
        self.filament_width_angstrom = (
            (filament_width_angstrom, filament_width_angstrom)
            if isinstance(filament_width_angstrom, (int, float))
            else filament_width_angstrom
        )
        self.native_to_render = self.render_size / self.image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, epoch_and_index: int | tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        if isinstance(epoch_and_index, tuple):
            epoch, index = epoch_and_index
        else:
            epoch, index = 0, epoch_and_index
        epoch = int(epoch)
        index = int(index)
        if epoch < 0 or index < 0:
            raise IndexError("epoch and sample index must be non-negative")

        # Allocate a separate 32-bit index namespace to every epoch. This keeps
        # samples reproducible without repeating epoch-zero images in later
        # epochs or colliding with the fixed validation seed range.
        sample_seed = self.seed + (epoch << 32) + index
        rng = np.random.default_rng(sample_seed)
        pixel_size_angstrom = float(rng.uniform(*self.pixel_size_angstrom))
        effective_pixel_size_angstrom = pixel_size_angstrom / self.native_to_render
        image, polylines, _ = self._generate(
            rng, pixel_size_angstrom=pixel_size_angstrom
        )
        polylines = [
            [
                (x * self.native_to_render, y * self.native_to_render)
                for x, y in polyline
            ]
            for polyline in polylines
        ]

        # Isolate stochastic transform state from model initialization and worker
        # scheduling, with epoch variation in the low bits for Torch's CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + epoch * len(self) + index)
            out = self.transform(
                image,
                polylines,
                pixel_size=effective_pixel_size_angstrom,
            )

        out["pixel_values"] = out.pop("image")
        valid = out["polylines"][:, :, 0].ne(-1).any(dim=1)
        out["class_labels"] = torch.full((len(valid),), -1, dtype=torch.long)
        out["class_labels"][valid] = 0
        out["image_id"] = torch.tensor(index, dtype=torch.long)
        input_height, input_width = out["pixel_values"].shape[-2:]
        out["input_size"] = torch.tensor([input_width, input_height], dtype=torch.long)
        out["native_size"] = torch.tensor(
            [self.image_size, self.image_size], dtype=torch.long
        )
        return out

    def _generate(
        self, rng: np.random.Generator, pixel_size_angstrom: float
    ) -> tuple[torch.Tensor, list[Polyline], list[float]]:
        native_size = self.image_size
        render_size = self.render_size
        native_to_render = self.native_to_render
        dark_background = bool(rng.integers(0, 2))
        background = 18 if dark_background else 237
        foreground = 237 if dark_background else 18
        image = Image.new("L", (render_size, render_size), color=background)
        draw = ImageDraw.Draw(image)

        if rng.random() < self.empty_probability:
            filament_count = 0
        else:
            filament_count = int(rng.integers(1, self.max_filaments + 1))

        polylines: list[Polyline] = []
        widths_angstrom: list[float] = []
        for filament_index in range(filament_count):
            if filament_index < 2 and filament_count >= 2:
                polyline = self._crossing_polyline(rng, filament_index)
            else:
                polyline = self._curved_polyline(rng)

            clipped = clip_polyline(polyline, (0, 0, native_size, native_size))
            if clipped is None:
                continue
            polylines.append(clipped)

            width_angstrom = float(rng.uniform(*self.filament_width_angstrom))
            width_native = width_angstrom / pixel_size_angstrom
            self._draw_filament(
                draw,
                clipped,
                foreground,
                width_native,
                coordinate_scale=native_to_render,
            )
            widths_angstrom.append(width_angstrom)

        blur_radius = float(rng.uniform(0.0, 2.5)) * native_to_render
        if blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        array = np.asarray(image, dtype=np.float32) / 255.0
        y_gradient = np.linspace(-1.0, 1.0, render_size, dtype=np.float32)[:, None]
        x_gradient = np.linspace(-1.0, 1.0, render_size, dtype=np.float32)[None, :]
        array += float(rng.uniform(-0.08, 0.08)) * x_gradient
        array += float(rng.uniform(-0.08, 0.08)) * y_gradient
        noise = rng.standard_normal(array.shape, dtype=np.float32)
        array += noise * rng.uniform(0.01, 0.08)

        # Contrast inversion is sampled independently of the initial black/white
        # background so both polarities occur through two deterministic routes.
        if rng.random() < 0.5:
            array = 1.0 - array
        array = np.clip(array, 1e-4, 1.0 - 1e-4)
        return torch.from_numpy(array).unsqueeze(0), polylines, widths_angstrom

    def _curved_polyline(self, rng: np.random.Generator) -> Polyline:
        size = self.image_size
        margin = size * 0.2
        start = rng.uniform(-margin, size + margin, size=2)
        end = rng.uniform(-margin, size + margin, size=2)
        while np.linalg.norm(end - start) < size * 0.25:
            end = rng.uniform(-margin, size + margin, size=2)

        direction = end - start
        normal = np.array([-direction[1], direction[0]])
        normal /= max(float(np.linalg.norm(normal)), 1e-6)
        bend = normal * rng.uniform(-0.3, 0.3) * np.linalg.norm(direction)
        control_1 = start + direction / 3 + bend
        control_2 = start + 2 * direction / 3 - bend * rng.uniform(0.2, 1.0)

        t = np.linspace(0.0, 1.0, 32, dtype=np.float64)[:, None]
        points = (
            (1 - t) ** 3 * start
            + 3 * (1 - t) ** 2 * t * control_1
            + 3 * (1 - t) * t**2 * control_2
            + t**3 * end
        )
        return [(float(x), float(y)) for x, y in points]

    def _crossing_polyline(
        self, rng: np.random.Generator, filament_index: int
    ) -> Polyline:
        size = self.image_size
        center = np.array([size / 2, size / 2]) + rng.uniform(
            -size * 0.08, size * 0.08, size=2
        )
        base_angle = rng.uniform(-0.2, 0.2)
        angle = base_angle + filament_index * np.pi / 2
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        distance = size * rng.uniform(0.65, 0.9)
        t = np.linspace(-distance, distance, 32)
        points = center + t[:, None] * direction
        points += (
            np.sin(np.linspace(-np.pi, np.pi, len(t)))[:, None]
            * normal
            * size
            * rng.uniform(0.01, 0.06)
        )
        return [(float(x), float(y)) for x, y in points]

    @staticmethod
    def _draw_filament(
        draw: ImageDraw.ImageDraw,
        polyline: Polyline,
        foreground: int,
        width_native: float,
        coordinate_scale: float = 1.0,
    ) -> None:
        points = [
            (round(x * coordinate_scale), round(y * coordinate_scale))
            for x, y in polyline
        ]
        width = max(1, round(width_native * coordinate_scale))

        # A narrow center stripe gives width/contrast structure rather than a
        # single binary stroke, while preserving the same centerline label.
        center = int((foreground + (18 if foreground > 128 else 237)) / 2)
        draw.line(points, fill=foreground, width=width, joint="curve")
        draw.line(
            points,
            fill=center,
            width=max(1, width // 4),
            joint="curve",
        )
