# Portions adapted from Hugging Face Transformers' Hungarian matcher:
# https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_for_object_detection.py
# Copyright 2024 The HuggingFace Team. All rights reserved.
# Modified for CryoTracer's polyline geometry and group matching.
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from cryotracer.model.loss import aspect_ratio_scale, frechet_distance

_DISTANCE_EPS = 1e-12


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        polyline_cost:
            The relative weight of the distance error of the polyline coordinates in the matching cost.
        distance_metric:
            The polyline matching distance ("frechet").
        queries_per_group:
            Number of queries in each independent group-DETR matching problem. If
            omitted, all queries are matched together as one group.
    """

    def __init__(
        self,
        class_cost: float = 1,
        polyline_cost: float = 1,
        distance_metric: str = "frechet",
        queries_per_group: int | None = None,
    ):
        super().__init__()

        self.class_cost = class_cost
        self.polyline_cost = polyline_cost
        if distance_metric != "frechet":
            raise ValueError("Only Fréchet matching is supported")
        self.distance_metric = frechet_distance
        if class_cost == 0 and polyline_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")
        if queries_per_group is not None and queries_per_group <= 0:
            raise ValueError("queries_per_group must be positive")
        self.queries_per_group = queries_per_group

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_polylines": Tensor of dim [batch_size, num_queries, num_points, 2] with the predicted polyline coordinates.
            targets (`list[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_polylines] (where num_target_polylines is the number of
                  ground-truth objects in the target) containing the class labels
                * "polylines": Tensor of dim [num_target_polylines, num_points, 2] containing the target polyline coordinates.
                * "image_size": Tensor containing (width, height) of the model input.

        Returns:
            `list[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = num_groups * min(
            queries_per_group, num_target_polylines
            ). Each group is matched independently against the same target set.
        """
        batch_size, num_queries = outputs["logits"].shape[:2]
        queries_per_group = self.queries_per_group or num_queries
        if num_queries % queries_per_group != 0:
            raise ValueError(
                f"Query count ({num_queries}) must be divisible by queries_per_group "
                f"({queries_per_group})"
            )
        if any(len(target["polylines"]) > queries_per_group for target in targets):
            raise ValueError("Target count exceeds num_queries")
        num_groups = num_queries // queries_per_group

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["logits"].float().flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_polylines = (
            outputs["pred_polylines"].float().flatten(0, 1)
        )  # [batch_size * num_queries, num_points, 2]

        # Also concat the target labels and polylines
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_polylines = torch.cat([v["polylines"] for v in targets])
        image_scales = aspect_ratio_scale(
            torch.stack([v["image_size"] for v in targets])
        ).to(device=out_polylines.device, dtype=out_polylines.dtype)
        out_scales = image_scales[:, None, None, :].expand_as(outputs["pred_polylines"])
        out_polylines = out_polylines * out_scales.flatten(0, 1)
        target_scales = torch.repeat_interleave(
            image_scales,
            torch.as_tensor(
                [len(v["polylines"]) for v in targets], device=image_scales.device
            ),
            dim=0,
        )
        target_polylines = target_polylines * target_scales[:, None, :]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        class_cost = -out_prob[:, target_ids]

        polyline_cost = (
            self.distance_metric(out_polylines, target_polylines)
            .clamp_min(_DISTANCE_EPS)
            .sqrt()
        )

        # Final cost matrix
        cost_matrix = self.polyline_cost * polyline_cost + self.class_cost * class_cost
        # cost_matrix shape: [batch_size * num_queries, total_targets]
        # Reshape to [batch_size, num_queries, targets_per_image]
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["polylines"]) for v in targets]
        indices = []
        for batch_index, batch_cost in enumerate(cost_matrix.split(sizes, -1)):
            image_cost = batch_cost[batch_index]
            source_indices = []
            target_indices = []
            for group_index in range(num_groups):
                group_start = group_index * queries_per_group
                group_cost = image_cost[group_start : group_start + queries_per_group]
                source, target = linear_sum_assignment(group_cost)
                source_indices.append(
                    torch.as_tensor(source, dtype=torch.int64) + group_start
                )
                target_indices.append(torch.as_tensor(target, dtype=torch.int64))
            indices.append(
                (
                    torch.cat(source_indices),
                    torch.cat(target_indices),
                )
            )
        return indices
