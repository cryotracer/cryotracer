# Portions adapted from Hugging Face Transformers' RF-DETR attention:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/rf_detr/modeling_rf_detr.py
# RF-DETR: https://github.com/roboflow/rf-detr
# Modified for CryoTracer to sample around every point of a polyline.
#
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# RF-DETR's deformable attention is derived from LW-DETR and Deformable DETR:
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
#
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn
from transformers.processing_utils import Unpack
from transformers.utils import torch_compilable_check
from transformers.utils.generic import TransformersKwargs

from cryotracer.model.network.configuration import PolylineConfig
from cryotracer.model.network.rf_detr_adapter import MultiScaleDeformableAttention


class PolylineMultiscaleDeformableAttention(nn.Module):
    """
    RF-DETR cross-attention with one full polyline reference per query.

    References have shape `(num_points, 2)`. Attention samples around each
    reference point and flattens those samples into RF-DETR's multiscale
    deformable attention kernel.
    """

    def __init__(self, config: PolylineConfig, num_heads: int, n_points: int):
        super().__init__()

        self.attn = MultiScaleDeformableAttention()

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got "
                f"{config.d_model} and {num_heads}"
            )

        self.im2col_step = 64
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.num_polyline_points = config.num_points
        self.n_points_per_polyline_point = n_points
        self.n_sampling_points = self.num_polyline_points * n_points

        self.sampling_offsets = nn.Linear(
            config.d_model,
            num_heads * self.n_levels * self.num_polyline_points * n_points * 2,
        )
        self.attention_weights = nn.Linear(
            config.d_model, num_heads * self.n_levels * self.n_sampling_points
        )
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
        torch_compilable_check(
            total_elements == sequence_length,
            "Make sure to align the spatial shapes with the sequence length of the encoder hidden states",
        )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(
            batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads
        )

        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.num_polyline_points,
            self.n_points_per_polyline_point,
            2,
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels * self.n_sampling_points,
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_sampling_points,
        )

        if reference_points.shape[-2:] != (self.num_polyline_points, 2):
            raise ValueError(
                "Polyline reference points must have shape "
                f"(..., {self.num_polyline_points}, 2), got {tuple(reference_points.shape)}"
            )

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]
        )
        sampling_locations = sampling_locations.flatten(4, 5)

        output = self.attn(
            value,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )

        output = self.output_proj(output)

        return output, attention_weights
