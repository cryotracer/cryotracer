# Portions adapted from Hugging Face Transformers' RF-DETR model:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/rf_detr/modeling_rf_detr.py
# RF-DETR: https://github.com/roboflow/rf-detr
# Modified for CryoTracer's polyline proposals, predictions, and losses.
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

import torch
from torch import Tensor, nn
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from cryotracer.model.loss import polyline_detection_loss
from cryotracer.model.network.backbone import PolylineFeaturePyramid
from cryotracer.model.network.base import PolylinePreTrainedModel
from cryotracer.model.network.configuration import PolylineConfig
from cryotracer.model.network.decoder import PolylineDecoder
from cryotracer.model.network.outputs import (
    PolylineEncoderDecoderOutput,
    PolylineOutput,
)
from cryotracer.model.network.rf_detr_adapter import (
    ConvEncoder,
    PredictionHead,
    gen_encoder_output_proposals,
    get_valid_ratio,
)


class PolylineEncoderDecoder(PolylinePreTrainedModel):
    get_valid_ratio = get_valid_ratio
    gen_encoder_output_proposals = gen_encoder_output_proposals

    def __init__(self, config: PolylineConfig):
        super().__init__(config)

        self.backbone = ConvEncoder(config)
        self.feature_pyramid = PolylineFeaturePyramid(config)

        self.group_detr = config.group_detr
        self.num_queries = config.num_queries
        hidden_dim = config.d_model
        self.reference_point_embed = nn.Embedding(
            self.num_queries * self.group_detr,
            config.num_points * 2,
        )
        self.query_feat = nn.Embedding(self.num_queries * self.group_detr, hidden_dim)

        self.decoder = PolylineDecoder(config)

        self.enc_output = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.group_detr)]
        )
        self.enc_output_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.group_detr)]
        )
        self.enc_out_polyline_embed = nn.ModuleList(
            [
                PredictionHead(
                    config.d_model,
                    config.d_model,
                    config.num_points * 2,
                    num_layers=3,
                )
                for _ in range(self.group_detr)
            ]
        )
        self.enc_out_class_embed = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.num_labels + 1)
                for _ in range(self.group_detr)
            ]
        )
        self.d_model = config.d_model

        self.post_init()

    def freeze_backbone(self) -> None:
        """Freeze only the core DINOv2 feature extractor.

        The RF-DETR scale projector intentionally remains trainable. The
        upstream RF-DETR method targets ``backbone.model``, which is not a
        member of the Transformers ``ConvEncoder`` used here.
        """
        for parameter in self.backbone.backbone.parameters():
            parameter.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.backbone.parameters():
            parameter.requires_grad_(True)

    def _flatten_feature_levels(
        self,
        feature_levels: list[Tensor],
        mask_levels: list[Tensor],
    ) -> tuple[Tensor, Tensor, list[tuple[int, int]], Tensor, Tensor, Tensor]:
        source_flatten = torch.cat(
            [feature.flatten(2).transpose(1, 2) for feature in feature_levels],
            dim=1,
        )
        mask_flatten = torch.cat([mask.flatten(1) for mask in mask_levels], dim=1)
        spatial_shapes_list = [feature.shape[-2:] for feature in feature_levels]
        spatial_shapes = torch.as_tensor(
            spatial_shapes_list, dtype=torch.long, device=source_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [
                self.get_valid_ratio(mask, dtype=source_flatten.dtype)
                for mask in mask_levels
            ],
            dim=1,
        )
        return (
            source_flatten,
            mask_flatten,
            spatial_shapes_list,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        )

    def _polyline_from_proposals(
        self, output_proposals: Tensor, polyline_delta: Tensor
    ) -> Tensor:
        polyline_delta = polyline_delta.reshape(
            *polyline_delta.shape[:-1], self.config.num_points, 2
        )
        centers = output_proposals[..., :2].unsqueeze(-2)
        sizes = output_proposals[..., 2:].unsqueeze(-2)
        polylines = centers + polyline_delta.tanh() * sizes
        return polylines.clamp(0, 1)

    def generate_topk_polyline_proposals(
        self,
        group_id: int,
        object_query_embedding: Tensor,
        output_proposals: Tensor,
        invalid_mask: Tensor,
        topk: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        object_query = self.enc_output[group_id](object_query_embedding)
        object_query = self.enc_output_norm[group_id](object_query)

        enc_outputs_class_proposals = self.enc_out_class_embed[group_id](object_query)
        polyline_delta = self.enc_out_polyline_embed[group_id](object_query)
        enc_outputs_class_proposals = enc_outputs_class_proposals.masked_fill(
            invalid_mask.to(enc_outputs_class_proposals.device), float("-inf")
        )

        enc_outputs_polyline = self._polyline_from_proposals(
            output_proposals, polyline_delta
        )
        foreground_scores = enc_outputs_class_proposals[
            ..., : self.config.num_labels
        ].max(-1)[0]
        topk_proposals = torch.topk(foreground_scores, topk, dim=1)[1]
        topk_polylines_undetach = torch.gather(
            enc_outputs_polyline,
            1,
            topk_proposals[:, :, None, None].expand(-1, -1, self.config.num_points, 2),
        )
        topk_polylines = topk_polylines_undetach.detach()
        object_query_undetach = torch.gather(
            object_query,
            1,
            topk_proposals.unsqueeze(-1).expand(-1, -1, self.config.d_model),
        )
        return object_query_undetach, topk_polylines, topk_polylines_undetach

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PolylineEncoderDecoderOutput:
        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(
                ((batch_size, height, width)), dtype=torch.long, device=device
            )

        features, mask = self.backbone(pixel_values, pixel_mask)
        feature_levels, mask_levels = self.feature_pyramid(features, mask)
        (
            source_flatten,
            mask_flatten,
            spatial_shapes_list,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        ) = self._flatten_feature_levels(feature_levels, mask_levels)

        object_query_embedding, output_proposals, invalid_mask = (
            self.gen_encoder_output_proposals(
                source_flatten, ~mask_flatten, spatial_shapes_list
            )
        )

        group_detr = self.group_detr if self.training else 1
        topk = self.num_queries
        topk_polylines = torch.empty(
            (batch_size, topk * group_detr, self.config.num_points, 2),
            device=device,
            dtype=output_proposals.dtype,
        )
        enc_outputs_polylines = torch.empty(
            (batch_size, topk * group_detr, self.config.num_points, 2),
            device=device,
            dtype=output_proposals.dtype,
        )
        enc_outputs_class = torch.empty(
            (batch_size, topk * group_detr, self.config.d_model),
            device=device,
            dtype=output_proposals.dtype,
        )
        for group_id in range(group_detr):
            (
                object_query_undetach,
                group_topk_polylines,
                topk_polylines_undetach,
            ) = self.generate_topk_polyline_proposals(
                group_id, object_query_embedding, output_proposals, invalid_mask, topk
            )
            group_slice = slice(group_id * topk, (group_id + 1) * topk)
            topk_polylines[:, group_slice] = group_topk_polylines
            enc_outputs_polylines[:, group_slice] = topk_polylines_undetach
            enc_outputs_class[:, group_slice] = object_query_undetach

        if self.training:
            reference_points = self.reference_point_embed.weight.sigmoid()
            query_feat = self.query_feat.weight
        else:
            reference_points = self.reference_point_embed.weight[
                : self.num_queries
            ].sigmoid()
            query_feat = self.query_feat.weight[: self.num_queries]

        reference_points = reference_points.reshape(-1, self.config.num_points, 2)
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1, -1)
        two_stage_len = enc_outputs_polylines.shape[-3]
        reference_points_two_stage_subset = topk_polylines
        reference_points_subset = reference_points[..., two_stage_len:, :, :]
        reference_points = torch.cat(
            [reference_points_two_stage_subset, reference_points_subset], dim=1
        )
        init_reference_points = reference_points
        target = query_feat.unsqueeze(0).expand(batch_size, -1, -1)

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
            **kwargs,
        )

        return PolylineEncoderDecoderOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            backbone_features=feature_levels,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_polylines=enc_outputs_polylines,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


class PolylineModel(PolylinePreTrainedModel):
    config_class = PolylineConfig

    _no_split_modules = None
    _tied_weights_keys = None

    def __init__(self, config: PolylineConfig):
        super().__init__(config)
        self.model = PolylineEncoderDecoder(config)
        self.class_embed = nn.Linear(config.d_model, config.num_labels + 1)

        self.loss_function = polyline_detection_loss

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PolylineOutput:
        outputs = self.model(pixel_values, pixel_mask=pixel_mask, **kwargs)

        hidden_states = outputs.intermediate_hidden_states
        intermediate_reference_points = outputs.intermediate_reference_points

        enc_outputs_class_logits = self.predict_encoder_class_logits(
            outputs.enc_outputs_class
        )
        outputs_class, outputs_coord = self.predict_class_and_polylines(
            hidden_states, intermediate_reference_points
        )
        logits = outputs_class[-1]
        pred_polylines = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs, indices = None, None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs, indices = self.loss_function(
                logits,
                labels,
                self.device,
                pred_polylines,
                self.config,
                outputs_class,
                outputs_coord,
                enc_outputs_class=enc_outputs_class_logits,
                enc_outputs_polylines=outputs.enc_outputs_polylines,
            )

        return PolylineOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_polylines=pred_polylines,
            auxiliary_outputs=auxiliary_outputs,
            indices=indices,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=enc_outputs_class_logits,
            enc_outputs_polylines=outputs.enc_outputs_polylines,
            backbone_features=outputs.backbone_features,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def predict_encoder_class_logits(self, enc_outputs_class: torch.Tensor) -> Tensor:
        enc_outputs_class_list = enc_outputs_class.split(self.config.num_queries, dim=1)
        group_detr = self.config.group_detr if self.training else 1
        pred_class = [
            self.model.enc_out_class_embed[group_index](
                enc_outputs_class_list[group_index]
            )
            for group_index in range(group_detr)
        ]
        return torch.cat(pred_class, dim=1)

    def predict_class_and_polylines(
        self, hidden_states: torch.Tensor, polyline_references: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.class_embed(hidden_states)
        return logits, polyline_references

    @classmethod
    def from_rf_detr_pretrained(
        cls, pretrained_model_name_or_path, *, config=None, **kwargs
    ):
        """Initialize from RF-DETR weights; use from_pretrained for native checkpoints."""
        from cryotracer.model.network.conversion import from_rf_detr_pretrained

        return from_rf_detr_pretrained(
            cls, pretrained_model_name_or_path, config=config, **kwargs
        )
