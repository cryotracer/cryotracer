from pathlib import Path
from urllib.parse import unquote

import lightning as L
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from cryotracer.model.network import PolylineConfig, PolylineModel

CHECKPOINT_FORMAT = "cryotracer-v1"


def resolve_checkpoint(source):
    """Resolve a local path or download a model checkpoint into the Hub cache."""
    if not str(source).startswith("hf://"):
        return Path(source).expanduser()
    parts = str(source)[5:].split("/", 2)
    if len(parts) != 3:
        raise ValueError("Expected hf://owner/repo[@revision]/file.ckpt")
    owner, repository, filename = parts
    repository, separator, revision = repository.partition("@")
    if (
        not owner
        or not repository
        or (separator and not revision)
        or any(part in {"", ".", ".."} for part in filename.split("/"))
    ):
        raise ValueError("Expected hf://owner/repo[@revision]/file.ckpt")
    try:
        return Path(
            hf_hub_download(
                repo_id=f"{owner}/{repository}",
                filename=filename,
                revision=unquote(revision) if separator else None,
            )
        )
    except HfHubHTTPError as error:
        raise OSError(f"Could not download checkpoint {source}: {error}") from error


def read_checkpoint(path):
    checkpoint = torch.load(
        resolve_checkpoint(path), map_location="cpu", weights_only=False
    )
    if checkpoint.get("cryotracer_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "Expected a checkpoint produced by cryotracer (old CryoTracer checkpoints are unsupported)"
        )
    return checkpoint


class PolylineModule(L.LightningModule):
    def __init__(
        self,
        model_config,
        preprocessing,
        learning_rate=1e-4,
        prediction_threshold=0.5,
        gradient_checkpointing=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PolylineModel(PolylineConfig(**model_config))
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @classmethod
    def from_checkpoint(cls, checkpoint, **overrides):
        module = cls(**(checkpoint["hyper_parameters"] | overrides))
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        return module

    @classmethod
    def from_rf_detr(cls, source, *, num_queries, num_points, preprocessing, **kwargs):
        source_config, _ = PolylineConfig.get_config_dict(source)
        source_config["id2label"] = {0: "filament"}
        source_config["label2id"] = {"filament": 0}
        backbone = dict(source_config["backbone_config"])
        backbone.update(num_channels=1, num_windows=1)
        config = PolylineConfig.from_dict(
            source_config,
            backbone_config=backbone,
            num_labels=1,
            num_queries=num_queries,
            num_points=num_points,
            group_detr=1,
            cross_attention_type="polyline",
            num_feature_levels=4,
            decoder_layers=6,
            auxiliary_loss=True,
            encoder_loss_coefficient=1.0,
            distance_metric="frechet",
        )
        network = PolylineModel.from_rf_detr_pretrained(source, config=config)
        module = cls(config.to_dict(), preprocessing, **kwargs)
        module.model.load_state_dict(network.state_dict(), strict=True)
        module.model.train()
        return module

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def _step(self, batch, stage):
        targets = []
        for points, labels, size in zip(
            batch["polylines"], batch["class_labels"], batch["input_size"]
        ):
            valid = labels.ne(-1)
            targets.append(
                {
                    "polylines": points[valid],
                    "class_labels": labels[valid],
                    "image_size": size,
                }
            )
        outputs = self.model(pixel_values=batch["pixel_values"], labels=targets)
        batch_size = len(targets)
        self.log(
            f"{stage}/loss",
            outputs.loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=stage == "val",
        )
        self.log_dict(
            {f"{stage}/{key}": value for key, value in outputs.loss_dict.items()},
            batch_size=batch_size,
            on_step=stage == "train",
            on_epoch=stage == "val",
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        backbone, other = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (backbone if "backbone" in name else other).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": self.hparams.learning_rate * 0.1},
                {"params": other},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["cryotracer_format"] = CHECKPOINT_FORMAT
