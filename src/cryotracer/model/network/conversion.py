# RF-DETR checkpoint mappings adapted from Hugging Face Transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/conversion_mapping.py
# Modified for CryoTracer's polyline model and grayscale input projection.
#
# Copyright (C) 2025 the HuggingFace Inc. team. All rights reserved.
#
# Upstream portions: Apache-2.0; CryoTracer modifications: GPL-3.0-only.
# SPDX-License-Identifier: GPL-3.0-only AND Apache-2.0

from typing import ClassVar

import torch
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.core_model_loading import (
    ConversionOps,
    WeightConverter,
    WeightRenaming,
)

from cryotracer.model.network.configuration import PolylineConfig


class CollapseInputChannels(ConversionOps):
    """Collapse pretrained RGB kernels while preserving replicated-gray output."""

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        model,
        full_layer_name: str,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        expected_shape = model.get_parameter(full_layer_name).shape

        if tensor.shape == expected_shape:
            converted = tensor
        elif (
            tensor.ndim == 4
            and len(expected_shape) == 4
            and expected_shape[1] == 1
            and tensor.shape[0] == expected_shape[0]
            and tensor.shape[2:] == expected_shape[2:]
        ):
            converted = tensor.sum(dim=1, keepdim=True)
        else:
            raise ValueError(
                f"Cannot convert input projection from {tuple(tensor.shape)} "
                f"to {tuple(expected_shape)}"
            )

        return {target_patterns[0]: converted}

    @property
    def reverse_op(self) -> "CollapseInputChannels":
        # Saved checkpoints already use the configured input-channel count.
        return CollapseInputChannels()


register_checkpoint_conversion_mapping(
    "_RfDetrImportModel",
    [
        WeightRenaming(source_patterns=["^"], target_patterns=["model."]),
        WeightRenaming(
            source_patterns=["model.backbone.0.encoder.encoder"],
            target_patterns=["model.backbone.backbone"],
        ),
        WeightRenaming(
            source_patterns=["model.backbone.0.projector"],
            target_patterns=["model.backbone.projector"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.cv1.conv"],
            target_patterns=["projector.projector_layer.conv1.conv"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.cv1.bn"],
            target_patterns=["projector.projector_layer.conv1.norm"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.cv2.conv"],
            target_patterns=["projector.projector_layer.conv2.conv"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.cv2.bn"],
            target_patterns=["projector.projector_layer.conv2.norm"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.1"],
            target_patterns=["projector.layer_norm"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.m.(\\d+).cv1.conv"],
            target_patterns=["projector.projector_layer.bottlenecks.\\1.conv1.conv"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.m.(\\d+).cv1.bn"],
            target_patterns=["projector.projector_layer.bottlenecks.\\1.conv1.norm"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.m.(\\d+).cv2.conv"],
            target_patterns=["projector.projector_layer.bottlenecks.\\1.conv2.conv"],
        ),
        WeightRenaming(
            source_patterns=["projector.stages.0.0.m.(\\d+).cv2.bn"],
            target_patterns=["projector.projector_layer.bottlenecks.\\1.conv2.norm"],
        ),
        WeightConverter(
            source_patterns=(
                "model.backbone.backbone.embeddings.patch_embeddings.projection.weight"
            ),
            target_patterns=(
                "model.backbone.backbone.embeddings.patch_embeddings.projection.weight"
            ),
            operations=[CollapseInputChannels()],
        ),
    ],
    overwrite=True,
)


def from_rf_detr_pretrained(model_class, source, *, config=None, **kwargs):
    # A dedicated loader class keeps RF-DETR key conversion out of native loads.
    class _RfDetrImportModel(model_class):
        _checkpoint_conversion_prefix_free = True
        _keys_to_ignore_on_load_missing: ClassVar[list] = [
            r"^model\.decoder\.",
            r"^model\.feature_pyramid\.",
            r"^model\.enc_output\.",
            r"^model\.enc_output_norm\.",
            r"^model\.enc_out_polyline_embed\.",
            r"^model\.enc_out_class_embed\.",
            r"^model\.reference_point_embed\.",
        ]
        _keys_to_ignore_on_load_unexpected: ClassVar[list] = [
            r"^bbox_embed\.",
            r"^model\.transformer\.",
            r"^model\.refpoint_embed\.",
        ]

    if config is None:
        config_kwargs = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "revision",
                "token",
                "local_files_only",
                "force_download",
            )
            if key in kwargs
        }
        config_dict, _ = PolylineConfig.get_config_dict(source, **config_kwargs)
        config = PolylineConfig.from_dict(config_dict)
    kwargs.setdefault("ignore_mismatched_sizes", True)
    model = _RfDetrImportModel.from_pretrained(source, config=config, **kwargs)
    if isinstance(model, tuple):
        loaded, info = model
        loaded.__class__ = model_class
        loaded._weight_conversions = []
        return loaded, info
    model.__class__ = model_class
    # Import conversion must not be reversed when saving a native checkpoint.
    model._weight_conversions = []
    return model
