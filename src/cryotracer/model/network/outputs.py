# Output structures adapted from Hugging Face Transformers' RF-DETR:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/rf_detr/modeling_rf_detr.py
# Modified for CryoTracer's polyline predictions and matching outputs.
#
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.utils import ModelOutput


@dataclass
class PolylineEncoderDecoderOutput(ModelOutput):
    init_reference_points: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    enc_outputs_class: Tensor | None = None
    enc_outputs_polylines: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    backbone_features: list[torch.Tensor] | None = None


@dataclass
class PolylineOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None
    logits: torch.FloatTensor | None = None
    pred_polylines: torch.FloatTensor | None = None
    auxiliary_outputs: list[dict] | None = None
    indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    init_reference_points: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    enc_outputs_class: Tensor | None = None
    enc_outputs_polylines: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    backbone_features: list[torch.Tensor] | None = None
