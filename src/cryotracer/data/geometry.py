import itertools

import numpy as np
import torch

Box = tuple[int, int, int, int]
Point = tuple[float, float]
Polyline = list[Point]


def clip_segment(
    p0: Point,
    p1: Point,
    box: Box,
) -> tuple[np.ndarray, np.ndarray] | None:
    xmin, ymin, xmax, ymax = box
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0

    for p, q in [(-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)]:
        if p == 0:
            if q < 0:
                return None
            continue

        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None

    a = np.array([x0 + t0 * dx, y0 + t0 * dy])
    b = np.array([x0 + t1 * dx, y0 + t1 * dy])
    return a, b


def clip_polyline(polyline: Polyline, box: Box) -> Polyline | None:
    points = np.asarray(polyline, dtype=float)
    clipped: Polyline = []

    for p0, p1 in itertools.pairwise(points):
        segment = clip_segment((p0[0], p0[1]), (p1[0], p1[1]), box)
        if segment is None:
            continue

        a, b = segment
        if not clipped or not np.allclose(clipped[-1], a):
            clipped.append((float(a[0]), float(a[1])))
        clipped.append((float(b[0]), float(b[1])))

    return clipped if len(clipped) >= 2 else None


def sample_polyline(polyline: Polyline, n: int) -> torch.Tensor:
    if n <= 0:
        return torch.empty((0, 2), dtype=torch.float32)

    points = torch.tensor(polyline, dtype=torch.float32)
    num_points = len(points)

    if num_points == 0:
        return torch.zeros((n, 2), dtype=torch.float32)
    if num_points == 1:
        return points.repeat(n, 1)

    if n <= num_points:
        if n == 1:
            return points[:1]

        keep_idx = torch.linspace(0, num_points - 1, n).long()
        return points[keep_idx]

    extra = n - num_points
    segments = points[1:] - points[:-1]
    segment_lengths = torch.norm(segments, dim=1)
    total_length = segment_lengths.sum()

    if total_length <= 1e-12:
        return points[-1:].repeat(n, 1)

    raw = segment_lengths / total_length * extra
    add_per_segment = torch.floor(raw).long()
    remainder = extra - int(add_per_segment.sum().item())
    if remainder > 0:
        order = torch.argsort(raw - add_per_segment.float(), descending=True)
        add_per_segment[order[:remainder]] += 1

    out: list[torch.Tensor] = [points[0]]
    for i in range(num_points - 1):
        k = int(add_per_segment[i].item())
        if k > 0:
            p0 = points[i]
            p1 = points[i + 1]
            for j in range(1, k + 1):
                t = j / (k + 1)
                out.append((1 - t) * p0 + t * p1)
        out.append(points[i + 1])

    return torch.stack(out, dim=0)
