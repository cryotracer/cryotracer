# Portions adapted from Hugging Face Transformers' RfDetrPreTrainedModel:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/rf_detr/modeling_rf_detr.py
# RF-DETR: https://github.com/roboflow/rf-detr
# Modified for CryoTracer's polyline attention and feature pyramid initialization.
#
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# RF-DETR's deformable attention initialization derives from LW-DETR and
# Deformable DETR:
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Copyright (c) 2020 SenseTime. All Rights Reserved.
#
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

import math
from typing import ClassVar

import torch
from torch import nn
from transformers import initialization as init
from transformers.modeling_utils import PreTrainedModel

from cryotracer.model.network.attention import (
    PolylineMultiscaleDeformableAttention,
)
from cryotracer.model.network.backbone import PolylineFeaturePyramid
from cryotracer.model.network.configuration import PolylineConfig
from cryotracer.model.network.rf_detr_adapter import (
    BoxDeformableAttention,
    GroupSelfAttention,
)


class PolylinePreTrainedModel(PreTrainedModel):
    config_class = PolylineConfig
    config: PolylineConfig
    supports_gradient_checkpointing = True
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules: ClassVar[list] = [
        r"RfDetrConvEncoder",
        r"PolylineDecoderLayer",
    ]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs: ClassVar[dict] = {
        "attentions": GroupSelfAttention,
        "cross_attentions": BoxDeformableAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, BoxDeformableAttention):
            init.constant_(module.sampling_offsets.weight, 0.0)
            thetas = torch.arange(module.n_heads, dtype=torch.int64).float() * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1

            init.copy_(module.sampling_offsets.bias, grid_init.view(-1))
            init.constant_(module.attention_weights.weight, 0.0)
            init.constant_(module.attention_weights.bias, 0.0)
            init.xavier_uniform_(module.value_proj.weight)
            init.constant_(module.value_proj.bias, 0.0)
            init.xavier_uniform_(module.output_proj.weight)
            init.constant_(module.output_proj.bias, 0.0)
        if hasattr(module, "level_embed"):
            init.normal_(module.level_embed)
        if hasattr(module, "refpoint_embed") and module.refpoint_embed is not None:
            init.constant_(module.refpoint_embed.weight, 0)
        if hasattr(module, "class_embed") and module.class_embed is not None:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            init.constant_(module.class_embed.bias, bias_value)
        if hasattr(module, "bbox_embed") and module.bbox_embed is not None:
            init.constant_(module.bbox_embed.layers[-1].weight, 0)
            init.constant_(module.bbox_embed.layers[-1].bias, 0)
        if hasattr(module, "segmentation_bias") and isinstance(
            module.segmentation_bias, nn.Parameter
        ):
            nn.init.constant_(module.segmentation_bias, 0.0)
        if isinstance(module, PolylineMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight, 0.0)
            thetas = torch.arange(
                module.n_heads,
                dtype=module.sampling_offsets.bias.dtype,
                device=module.sampling_offsets.bias.device,
            ) * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
            grid_init = grid_init.view(module.n_heads, 1, 1, 1, 2).repeat(
                1,
                module.n_levels,
                module.num_polyline_points,
                module.n_points_per_polyline_point,
                1,
            )
            for point_idx in range(module.n_points_per_polyline_point):
                grid_init[:, :, :, point_idx, :] *= point_idx + 1

            module.sampling_offsets.bias.copy_(grid_init.reshape(-1))
            nn.init.constant_(module.attention_weights.weight, 0.0)
            nn.init.constant_(module.attention_weights.bias, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight)
            nn.init.constant_(module.value_proj.bias, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight)
            nn.init.constant_(module.output_proj.bias, 0.0)

        if isinstance(module, PolylineFeaturePyramid):
            for projection in module.projections:
                nn.init.xavier_uniform_(projection[0].weight)
                nn.init.constant_(projection[0].bias, 0.0)
