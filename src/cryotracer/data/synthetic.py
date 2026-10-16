from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from cryotracer.data.geometry import Polyline, clip_polyline


class SyntheticFilamentDataset(Dataset):
    """Deterministic-per-epoch full-frame synthetic filament micrographs.

    A sample is a pure function of ``(seed, epoch, index)``, giving reproducible
    runs with fresh training samples every epoch. Appearance variants share
    curves, widths, and augmentations. Plain integer indices use epoch zero so
    validation samples remain fixed.
    """

    def __init__(
        self,
        transform: Callable,
        image_size: int = 4096,
        num_samples: int = 128,
        seed: int = 42,
        max_filaments: int = 40,
        empty_probability: float = 0.1,
        pixel_size_angstrom: float | tuple[float, float] = (0.65, 1.5),
        render_size: int | None = None,
        filament_width_angstrom: float | tuple[float, float] = (100.0, 200.0),
        contrast_to_background_sigma: tuple[float, float] = (0.2, 2.0),
        easy_filament_probability: float = 0.1,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if max_filaments <= 0:
            raise ValueError("max_filaments must be positive")
        if not 0 <= empty_probability <= 1:
            raise ValueError("empty_probability must be between 0 and 1")
        if not 0 <= easy_filament_probability <= 1:
            raise ValueError("easy_filament_probability must be between 0 and 1")
        if (
            len(contrast_to_background_sigma) != 2
            or not np.isfinite(contrast_to_background_sigma).all()
            or not 0
            < contrast_to_background_sigma[0]
            <= contrast_to_background_sigma[1]
        ):
            raise ValueError(
                "contrast_to_background_sigma must be a positive ordered range"
            )
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
        self.contrast_to_background_sigma = contrast_to_background_sigma
        self.easy_filament_probability = easy_filament_probability

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
        # Keep geometry and widths independent of contrast settings.
        extra = deepcopy(rng)
        geometry_rng, contrast_rng, background_rng = [
            np.random.default_rng(seed)
            for seed in extra.integers(0, np.iinfo(np.int64).max, size=3)
        ]
        dark_background = bool(rng.integers(0, 2))
        background = 18 if dark_background else 237
        foreground = 237 if dark_background else 18
        image = Image.new("L", (render_size, render_size), color=background)
        draw = ImageDraw.Draw(image)

        if rng.random() < self.empty_probability:
            filament_count = 0
        else:
            filament_count = int(rng.integers(1, self.max_filaments + 1))

        if filament_count:
            # Mix sparse and denser scenes; counts above 30 remain rare.
            if self.max_filaments > 30 and geometry_rng.random() < 0.05:
                filament_count = int(geometry_rng.integers(31, self.max_filaments + 1))
            else:
                cap = min(self.max_filaments, 30)
                if cap <= 10 or geometry_rng.random() < 0.5:
                    filament_count = int(geometry_rng.integers(1, min(cap, 10) + 1))
                else:
                    filament_count = int(geometry_rng.integers(11, cap + 1))
            direction = float(geometry_rng.uniform(0, 2 * np.pi))
            aligned = geometry_rng.random() < 0.75
            angle_sigma = float(geometry_rng.uniform(0.04, 0.25))
            crossing_scene = filament_count >= 2 and geometry_rng.random() < 0.1
            crossing_center = geometry_rng.uniform(0.15, 0.85, size=2) * native_size
            crossing_angle = float(geometry_rng.uniform(np.pi / 18, np.pi / 2))
            crossing_half_length = native_size * float(geometry_rng.uniform(0.12, 0.4))

        polylines: list[Polyline] = []
        widths_angstrom: list[float] = []
        for filament_index in range(filament_count):
            if crossing_scene and filament_index < 2:
                angle = direction + filament_index * crossing_angle
                vector = np.array([np.cos(angle), np.sin(angle)])
                positions = np.linspace(-crossing_half_length, crossing_half_length, 32)
                polyline = (crossing_center + positions[:, None] * vector).tolist()
            else:
                angle = (
                    direction + float(geometry_rng.normal(0, angle_sigma))
                    if aligned
                    else float(geometry_rng.uniform(0, 2 * np.pi))
                )
                polyline = self._visible_polyline(
                    geometry_rng, angle, pixel_size_angstrom
                )

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
        background_field = float(rng.uniform(-0.08, 0.08)) * x_gradient
        background_field = (
            background_field + float(rng.uniform(-0.08, 0.08)) * y_gradient
        )
        noise = rng.standard_normal(array.shape, dtype=np.float32)
        noise_sigma = rng.uniform(0.01, 0.08)
        colored = gaussian_filter(noise, sigma=float(background_rng.uniform(0.5, 2)))
        colored /= max(float(colored.std()), 1e-6)
        noise = (noise + colored) / 2
        noise /= max(float(noise.std()), 1e-6)
        coarse = background_rng.standard_normal((8, 8), dtype=np.float32)
        ice = np.asarray(
            Image.fromarray(coarse).resize(
                (render_size, render_size), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ).copy()
        ice -= ice.mean()
        ice /= max(float(ice.std()), 1e-6)
        background_field += float(background_rng.uniform(0, 0.04)) * ice
        background_field += noise * noise_sigma
        if contrast_rng.random() < self.easy_filament_probability:
            ratio = float(contrast_rng.uniform(3, 8))
        else:
            ratio = float(
                np.exp(contrast_rng.uniform(*np.log(self.contrast_to_background_sigma)))
            )
        scale = (
            ratio * float(background_field.std()) / (abs(foreground - background) / 255)
        )
        # Center the intensity range so weak signal and noise are not
        # clipped against the original near-black/near-white background.
        array = 0.5 + (array - background / 255) * scale
        array += background_field

        # Contrast inversion is sampled independently of the initial black/white
        # background so both polarities occur through two deterministic routes.
        if rng.random() < 0.5:
            array = 1.0 - array
        array = np.clip(array, 1e-4, 1.0 - 1e-4)
        return torch.from_numpy(array).unsqueeze(0), polylines, widths_angstrom

    def _visible_polyline(
        self, rng: np.random.Generator, angle: float, pixel_size_angstrom: float
    ) -> Polyline:
        """Sample arc length in physical units and fit one complete trace in-frame."""
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        t = np.linspace(0, 1, 32)
        if rng.random() < 0.7:
            bend = float(rng.uniform(-0.005, 0.005))
        else:
            upper = 0.5 if rng.random() < 0.1 else 0.12
            bend = float(rng.uniform(0.015, upper)) * (1 if rng.random() < 0.5 else -1)
        points = (t - 0.5)[:, None] * direction + (bend * np.sin(np.pi * t))[
            :, None
        ] * normal
        points /= np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        # Long-length bins are conditional on the field of view and orientation.
        # Fitting the entire curve also avoids joining separated clipped pieces.
        max_length_nm = (
            self.image_size
            / np.ptp(points, axis=0).max()
            * pixel_size_angstrom
            / 10
            * 0.98
        )
        lower = np.array([3.0, 50.0, 300.0])
        upper = np.minimum([50.0, 300.0, 650.0], max_length_nm)
        weights = np.array([0.1, 0.65, 0.25]) * (upper > lower)
        if weights.sum() == 0:
            length_nm = float(rng.uniform(max_length_nm * 0.1, max_length_nm))
        else:
            bucket = int(rng.choice(3, p=weights / weights.sum()))
            length_nm = float(rng.uniform(lower[bucket], upper[bucket]))
        points *= length_nm * 10 / pixel_size_angstrom
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        translation = rng.uniform(-minimum, self.image_size - maximum)
        if rng.random() < 0.65:
            # Attach an endpoint to a border when it is an extremum of the curve.
            choices = []
            for endpoint in points[[0, -1]]:
                for axis in (0, 1):
                    if np.isclose(endpoint[axis], minimum[axis], atol=1e-8, rtol=0):
                        choices.append((axis, -minimum[axis]))
                    if np.isclose(endpoint[axis], maximum[axis], atol=1e-8, rtol=0):
                        choices.append((axis, self.image_size - maximum[axis]))
            if choices:
                axis, value = choices[int(rng.integers(len(choices)))]
                translation[axis] = value
        points += translation
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
