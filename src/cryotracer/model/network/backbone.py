# Feature pyramid inspired by Hugging Face Transformers' Deformable DETR:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/deformable_detr/modeling_deformable_detr.py
# Modified for CryoTracer to extend RF-DETR's projected feature map.
#
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
#
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

import torch
import torch.nn.functional as F
from torch import nn

from cryotracer.model.network.configuration import PolylineConfig


class PolylineFeaturePyramid(nn.Module):
    def __init__(self, config: PolylineConfig):
        super().__init__()
        self.num_feature_levels = config.num_feature_levels
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        config.d_model,
                        config.d_model,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(32, config.d_model),
                )
                for _ in range(max(0, config.num_feature_levels - 1))
            ]
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        feature_levels = [features]
        mask_levels = [mask]

        current = features
        for projection in self.projections:
            current = projection(current)
            feature_levels.append(current)
            mask_levels.append(
                F.interpolate(mask[None].float(), size=current.shape[-2:]).to(
                    torch.bool
                )[0]
            )

        return feature_levels, mask_levels
