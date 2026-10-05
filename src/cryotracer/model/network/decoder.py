# Portions adapted from Hugging Face Transformers' RF-DETR decoder:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/rf_detr/modeling_rf_detr.py
# RF-DETR: https://github.com/roboflow/rf-detr
# Modified for CryoTracer's polyline references and iterative refinement.
#
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# RF-DETR's transformer is derived from LW-DETR, Conditional DETR, and DETR:
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

from typing import ClassVar

import torch
from torch import Tensor, nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from cryotracer.model.network.attention import (
    PolylineMultiscaleDeformableAttention,
)
from cryotracer.model.network.base import PolylinePreTrainedModel
from cryotracer.model.network.configuration import PolylineConfig
from cryotracer.model.network.rf_detr_adapter import (
    BoxDeformableAttention,
    DecoderMLP,
    GroupSelfAttention,
    PolylineDecoderOutput,
    PredictionHead,
    encode_sinusoidal_position_embedding,
    get_box_reference,
)


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class PolylineDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PolylineConfig, layer_idx: int):
        nn.Module.__init__(self)

        self.self_attn = GroupSelfAttention(config, layer_idx=layer_idx)
        self.dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)

        cross_attention_class = (
            BoxDeformableAttention
            if config.cross_attention_type == "original"
            else PolylineMultiscaleDeformableAttention
        )
        self.cross_attn = cross_attention_class(
            config,
            num_heads=config.decoder_cross_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model)

        self.mlp = DecoderMLP(config)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        self_attention_output, _ = self.self_attn(
            hidden_states, position_embeddings=position_embeddings, **kwargs
        )

        self_attention_output = nn.functional.dropout(
            self_attention_output, p=self.dropout, training=self.training
        )
        hidden_states = hidden_states + self_attention_output
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attention_output, _ = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )
        cross_attention_output = nn.functional.dropout(
            cross_attention_output, p=self.dropout, training=self.training
        )
        hidden_states = hidden_states + cross_attention_output
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class PolylineDecoder(PolylinePreTrainedModel):
    _can_record_outputs: ClassVar[dict] = {
        "hidden_states": PolylineDecoderLayer,
        "attentions": OutputRecorder(
            GroupSelfAttention, layer_name="self_attn", index=1
        ),
        "cross_attentions": [
            OutputRecorder(
                PolylineMultiscaleDeformableAttention,
                layer_name="cross_attn",
                index=1,
            ),
            OutputRecorder(
                BoxDeformableAttention,
                layer_name="cross_attn",
                index=1,
            ),
        ],
    }

    def __init__(self, config: PolylineConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [PolylineDecoderLayer(config, i) for i in range(config.decoder_layers)]
        )
        self.layernorm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        reference_coordinates = (
            4 if config.cross_attention_type == "original" else config.num_points * 2
        )
        self.ref_point_head = PredictionHead(
            reference_coordinates * (config.d_model // 2),
            config.d_model,
            config.d_model,
            num_layers=2,
        )
        self.polyline_refine_embed = nn.ModuleList(
            [
                PredictionHead(
                    config.d_model,
                    config.d_model,
                    config.num_points * 2,
                    num_layers=3,
                )
                for _ in range(config.decoder_layers)
            ]
        )

        self.post_init()

    def get_reference(
        self, reference_points: Tensor, valid_ratios: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.config.cross_attention_type == "original":
            # Keep polyline refinement and supervision, but feed the enclosing
            # (cx, cy, width, height) box into RF-DETR's original attention path.
            lower = reference_points.amin(dim=-2)
            upper = reference_points.amax(dim=-2)
            boxes = torch.cat(((lower + upper) / 2, upper - lower), dim=-1)
            return get_box_reference(self, boxes, valid_ratios)

        reference_points_inputs = (
            reference_points[:, :, None] * valid_ratios[:, None, :, None]
        )
        query_sine_embed = encode_sinusoidal_position_embedding(
            reference_points.flatten(-2),
            num_pos_feats=self.config.d_model // 2,
        )
        query_pos = self.ref_point_head(query_sine_embed)
        return reference_points_inputs, query_pos

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        valid_ratios: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        intermediate = ()

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        intermediate_reference_points = ()
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_inputs, query_pos = self.get_reference(
                reference_points, valid_ratios
            )
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_embeddings=query_pos,
                reference_points=reference_points_inputs,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                **kwargs,
            )
            hidden_states = self.layernorm(hidden_states)

            polyline_delta = self.polyline_refine_embed[idx](hidden_states).reshape(
                *hidden_states.shape[:-1],
                self.config.num_points,
                2,
            )
            refined_reference_points = (
                inverse_sigmoid(reference_points) + polyline_delta
            ).sigmoid()
            reference_points = refined_reference_points.detach()

            intermediate += (hidden_states,)
            intermediate_reference_points += (refined_reference_points,)

        intermediate = torch.stack(intermediate)
        last_hidden_state = intermediate[-1]
        intermediate_reference_points = torch.stack(intermediate_reference_points)

        return PolylineDecoderOutput(
            last_hidden_state=last_hidden_state,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
        )
