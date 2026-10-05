# Portions adapted from Hugging Face Transformers' detection loss:
# https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_for_object_detection.py
# Copyright 2024 The HuggingFace Team. All rights reserved.
# Modified for CryoTracer's polyline geometry and group matching.
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

import torch
import torch.distributed as dist
from transformers.loss.loss_for_object_detection import ImageLoss


def _polyline_pairs(
    pred_polylines: torch.Tensor,
    target_polylines: torch.Tensor,
    *,
    aligned: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if aligned:
        if pred_polylines.shape[0] != target_polylines.shape[0]:
            raise ValueError("Aligned distances require equal polyline counts")
        return pred_polylines, target_polylines
    return pred_polylines[:, None], target_polylines[None, :]


def aspect_ratio_scale(image_size: torch.Tensor) -> torch.Tensor:
    """Return an ``(x, y)`` scale that preserves image-space geometry.

    Polyline coordinates are normalized independently by image width and height.
    Dividing both dimensions by the longer side makes distances isotropic for
    rectangular images while preserving the existing scale for square images.
    """
    if image_size.shape[-1] != 2:
        raise ValueError("image_size must contain (width, height)")
    if torch.any(image_size <= 0):
        raise ValueError("image_size values must be positive")

    image_size = image_size.to(dtype=torch.float32)
    return image_size / image_size.amax(dim=-1, keepdim=True)


def frechet_distance(
    pred_polylines: torch.Tensor,
    target_polylines: torch.Tensor,
    *,
    aligned: bool = False,
) -> torch.Tensor:
    """
    Squared discrete Fréchet distance, best of both GT orientations.

    Uses the standard DP recurrence:
        dp[i][j] = max(D[i][j], min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))

    Args:
        pred_polylines: (Q, Np, 2) predicted polylines
        target_polylines:   (G, Ng, 2) target polylines
        aligned: Compute only corresponding pairs; requires Q == G.

    Returns:
        (Q, G) pairwise or (Q,) aligned squared Fréchet distances
    """
    Np = pred_polylines.shape[1]
    Ng = target_polylines.shape[1]

    pred, target = _polyline_pairs(pred_polylines, target_polylines, aligned=aligned)
    diff = pred[..., :, None, :] - target[..., None, :, :]
    D = (diff * diff).sum(-1)

    def _frechet_dp(D: torch.Tensor) -> torch.Tensor:
        """DP over vertex dimensions, preserving the leading pair dimensions."""
        # Build first row via cumulative max
        row = [D[..., 0, 0]]
        for j in range(1, Ng):
            row.append(torch.maximum(row[-1], D[..., 0, j]))
        dp_prev = torch.stack(row, dim=-1)

        for i in range(1, Np):
            row = [torch.maximum(dp_prev[..., 0], D[..., i, 0])]
            for j in range(1, Ng):
                prev_min = torch.minimum(
                    dp_prev[..., j],
                    torch.minimum(row[-1], dp_prev[..., j - 1]),
                )
                row.append(torch.maximum(D[..., i, j], prev_min))
            dp_prev = torch.stack(row, dim=-1)

        return dp_prev[..., -1]

    fwd = _frechet_dp(D)
    rev = _frechet_dp(D.flip(-1))  # reverse GT orientation

    return torch.minimum(fwd, rev)


_DISTANCE_EPS = 1e-12


class PolylineImageLoss(ImageLoss):
    """
    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`list[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
        distance_metric (`str`):
            Polyline loss: "frechet". The distance returns squared values; this
            criterion takes their square root before averaging over matches.
    """

    def __init__(
        self,
        matcher,
        num_classes,
        eos_coef,
        losses,
        distance_metric="frechet",
        queries_per_group: int | None = None,
    ):
        super().__init__(matcher, num_classes, eos_coef, losses)
        self.queries_per_group = queries_per_group

        if distance_metric != "frechet":
            raise ValueError("Only Fréchet loss is supported")
        self.distance_metric = frechet_distance

    def forward(self, outputs, targets):
        """Compute DETR losses with target-count normalization across DDP ranks."""
        outputs_without_aux = {
            key: value for key, value in outputs.items() if key != "auxiliary_outputs"
        }
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = torch.as_tensor(
            [sum(len(target["class_labels"]) for target in targets)],
            dtype=torch.float32,
            device=outputs_without_aux["logits"].device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_boxes)
            num_boxes /= dist.get_world_size()
        normalized_num_boxes = num_boxes.clamp_min(1).item()

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs_without_aux,
                    targets,
                    indices,
                    normalized_num_boxes,
                )
            )

        for index, auxiliary_outputs in enumerate(outputs.get("auxiliary_outputs", [])):
            auxiliary_indices = self.matcher(auxiliary_outputs, targets)
            for loss in self.losses:
                if loss == "masks":
                    continue
                auxiliary_losses = self.get_loss(
                    loss,
                    auxiliary_outputs,
                    targets,
                    auxiliary_indices,
                    normalized_num_boxes,
                )
                losses.update(
                    {
                        key + f"_{index}": value
                        for key, value in auxiliary_losses.items()
                    }
                )

        return losses

    def loss_polylines(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the polylines.

        Targets dicts must contain "polylines" with shape
        [nb_target_polylines, num_points, 2] and "image_size" containing
        (width, height). The target polylines are expected in format (x, y),
        normalized by the corresponding image dimensions.
        """
        if "pred_polylines" not in outputs:
            raise KeyError("No predicted polylines found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_polylines = outputs["pred_polylines"][idx].float()
        target_polylines = torch.cat(
            [t["polylines"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        matched_scales = torch.cat(
            [
                aspect_ratio_scale(t["image_size"])[None].expand(len(i), -1)
                for t, (_, i) in zip(targets, indices)
            ],
            dim=0,
        ).to(device=source_polylines.device, dtype=source_polylines.dtype)
        source_polylines = source_polylines * matched_scales[:, None, :]
        target_polylines = target_polylines * matched_scales[:, None, :]

        loss_polyline = (
            self.distance_metric(source_polylines, target_polylines, aligned=True)
            .clamp_min(_DISTANCE_EPS)
            .sqrt()
        )

        num_queries = outputs["pred_polylines"].shape[1]
        queries_per_group = self.queries_per_group or num_queries
        if num_queries % queries_per_group != 0:
            raise ValueError(
                f"Query count ({num_queries}) must be divisible by queries_per_group "
                f"({queries_per_group})"
            )
        num_groups = num_queries // queries_per_group

        losses = {}
        losses["loss_polyline"] = loss_polyline.sum() / (num_boxes * num_groups)

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "polylines": self.loss_polylines,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)


def _set_aux_loss(outputs_class, outputs_coord):
    return [
        {"logits": a, "pred_polylines": b}
        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
    ]


def polyline_detection_loss(
    logits,
    labels,
    device,
    pred_polylines,
    config,
    outputs_class=None,
    outputs_coord=None,
    enc_outputs_class=None,
    enc_outputs_polylines=None,
    **kwargs,
):
    from cryotracer.model.matcher import HungarianMatcher

    # First: create the matcher
    matcher = HungarianMatcher(
        class_cost=config.class_cost,
        polyline_cost=config.polyline_cost,
        distance_metric=config.distance_metric,
        queries_per_group=config.num_queries,
    )
    # Second: create the criterion
    losses = ["labels", "polylines"]
    criterion = PolylineImageLoss(
        matcher=matcher,
        num_classes=config.num_labels,
        eos_coef=config.eos_coefficient,
        losses=losses,
        distance_metric=config.distance_metric,
        queries_per_group=config.num_queries,
    )
    criterion.to(device)
    # Third: compute the losses, based on outputs and labels
    outputs_loss = {}
    auxiliary_outputs = None
    outputs_loss["logits"] = logits
    outputs_loss["pred_polylines"] = pred_polylines
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs

    loss_dict = criterion(outputs_loss, labels)
    if enc_outputs_class is not None and enc_outputs_polylines is not None:
        enc_outputs_loss = {
            "logits": enc_outputs_class,
            "pred_polylines": enc_outputs_polylines,
        }
        enc_loss_dict = criterion(enc_outputs_loss, labels)
        loss_dict.update({k + "_enc": v for k, v in enc_loss_dict.items()})

    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_polyline": config.polyline_loss_coefficient}
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    if enc_outputs_class is not None and enc_outputs_polylines is not None:
        encoder_loss_coefficient = getattr(config, "encoder_loss_coefficient", 1.0)
        weight_dict.update(
            {
                "loss_ce_enc": encoder_loss_coefficient,
                "loss_polyline_enc": (
                    config.polyline_loss_coefficient * encoder_loss_coefficient
                ),
            }
        )
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

    # TODO: reuse from above
    outputs_without_aux = {
        k: v for k, v in outputs_loss.items() if k != "auxiliary_outputs"
    }
    # Retrieve the matching between the outputs of the last layer and the targets
    indices = matcher(outputs_without_aux, labels)

    return loss, loss_dict, auxiliary_outputs, indices
