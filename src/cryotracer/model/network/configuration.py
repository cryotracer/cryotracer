# Configuration adapted from Hugging Face Transformers' RfDetrConfig:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/rf_detr/configuration_rf_detr.py
# RF-DETR is developed by Roboflow: https://github.com/roboflow/rf-detr
# Modified for CryoTracer's polyline detection and matching configuration.
#
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

from typing import ClassVar

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig

from cryotracer.model.network.rf_detr_adapter import PolylineBackboneConfig


@strict
class PolylineConfig(PreTrainedConfig):
    """CryoTracer configuration, for the RF-DETR polyline network."""

    model_type = "polyline_rf_detr"
    sub_configs: ClassVar[dict] = {"backbone_config": PolylineBackboneConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_expansion: float = 0.5
    c2f_num_blocks: int = 3
    activation_function: str = "silu"
    dropout: float = 0.1
    decoder_ffn_dim: int = 2048
    decoder_n_points: int = 4
    decoder_layers: int = 3
    decoder_self_attention_heads: int = 8
    decoder_cross_attention_heads: int = 16
    decoder_activation_function: str = "relu"
    num_queries: int = 10
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    group_detr: int = 1
    init_std: float = 0.02
    disable_custom_kernels: bool = True
    class_cost: int | float = 2
    bbox_cost: int | float = 5
    giou_cost: int | float = 2
    class_loss_coefficient: int | float = 1
    dice_loss_coefficient: int | float = 1
    bbox_loss_coefficient: int | float = 5
    giou_loss_coefficient: int | float = 2
    eos_coefficient: float = 0.1
    focal_alpha: float = 0.25
    auxiliary_loss: bool = True
    d_model: int = 256

    layer_norm_eps: float = 1e-5
    num_feature_levels: int = 1
    mask_loss_coefficient: int | float = 1
    mask_point_sample_ratio: int = 16
    mask_downsample_ratio: int = 4
    mask_class_loss_coefficient: int | float = 5.0
    mask_dice_loss_coefficient: int | float = 5.0
    segmentation_head_activation_function: str = "gelu"
    intermediate_size: int = 1024

    num_points: int = 3
    distance_metric: str = "frechet"
    polyline_cost: float = 2.0
    polyline_loss_coefficient: float = 10.0
    encoder_loss_coefficient: float = 1.0
    cross_attention_type: str = "polyline"

    def __post_init__(self, **kwargs):
        if self.cross_attention_type not in {"polyline", "original"}:
            raise ValueError(
                "cross_attention_type must be 'polyline' or 'original', "
                f"got {self.cross_attention_type!r}"
            )
        backbone = self.backbone_config
        if backbone is None:
            backbone = {
                "num_attention_heads": 6,
                "out_features": ["stage2", "stage5", "stage8", "stage11"],
                "hidden_size": 384,
                "num_register_tokens": 0,
                "image_size": 518,
            }
        elif isinstance(backbone, PreTrainedConfig):
            backbone = backbone.to_dict()
        backbone = dict(backbone)
        backbone_type = backbone.pop("model_type", "rf_detr_dinov2")
        if backbone_type not in {"rf_detr_dinov2"}:
            raise ValueError(f"Unsupported polyline backbone: {backbone_type!r}")
        self.backbone_config = PolylineBackboneConfig(**backbone)
        super().__post_init__(**kwargs)
