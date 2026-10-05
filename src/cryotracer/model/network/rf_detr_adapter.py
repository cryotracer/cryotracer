# Reuses RF-DETR components from Hugging Face Transformers by import:
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/rf_detr
# RF-DETR: https://github.com/roboflow/rf-detr
# Upstream notices: Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Upstream license: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0).

from transformers import RfDetrDinov2Config as PolylineBackboneConfig
from transformers.models.rf_detr.modeling_rf_detr import (
    MultiScaleDeformableAttention,
    RfDetrDecoder,
    RfDetrModel,
    encode_sinusoidal_position_embedding,
)
from transformers.models.rf_detr.modeling_rf_detr import (
    RfDetrAttention as GroupSelfAttention,
)
from transformers.models.rf_detr.modeling_rf_detr import (
    RfDetrConvEncoder as ConvEncoder,
)
from transformers.models.rf_detr.modeling_rf_detr import (
    RfDetrDecoderOutput as PolylineDecoderOutput,
)
from transformers.models.rf_detr.modeling_rf_detr import (
    RfDetrMLP as DecoderMLP,
)
from transformers.models.rf_detr.modeling_rf_detr import (
    RfDetrMLPPredictionHead as PredictionHead,
)
from transformers.models.rf_detr.modeling_rf_detr import (
    RfDetrMultiscaleDeformableAttention as BoxDeformableAttention,
)

get_box_reference = RfDetrDecoder.get_reference
get_valid_ratio = RfDetrModel.get_valid_ratio
gen_encoder_output_proposals = RfDetrModel.gen_encoder_output_proposals

__all__ = [
    "BoxDeformableAttention",
    "ConvEncoder",
    "DecoderMLP",
    "GroupSelfAttention",
    "MultiScaleDeformableAttention",
    "PolylineBackboneConfig",
    "PolylineDecoderOutput",
    "PredictionHead",
    "encode_sinusoidal_position_embedding",
    "gen_encoder_output_proposals",
    "get_box_reference",
    "get_valid_ratio",
]
