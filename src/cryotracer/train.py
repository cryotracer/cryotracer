import argparse
import json
import logging
from copy import copy
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Subset

from cryotracer.data.cbox import CBoxDataset
from cryotracer.data.datamodule import DataModule
from cryotracer.data.parquet import ParquetDataset
from cryotracer.data.synthetic import SyntheticFilamentDataset
from cryotracer.data.transforms import PolylineTransform
from cryotracer.model.module import PolylineModule, read_checkpoint

logger = logging.getLogger(__name__)

SYNTHETIC_OPTIONS = (
    "image_size",
    "train_samples",
    "val_samples",
    "max_filaments",
    "empty_probability",
    "pixel_size_angstrom",
    "filament_width_angstrom",
)


def runtime(device):
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    precision = (
        "bf16-mixed"
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else "32-true"
    )
    return device, precision


def configure_logging(output_dir, command):
    handler = logging.FileHandler(output_dir / f"{command}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return handler


def _plain(value):
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, tuple):
        return list(value)
    return value


def resolve_run(args):
    config = {
        key: _plain(value) for key, value in vars(args).items() if key != "handler"
    }
    if config.get("input"):
        config["input"] = str(Path(config["input"]).expanduser().absolute())
    return config


def build_data(run, model_config, preprocessing):
    common = {
        "num_queries": model_config.num_queries,
        "num_points": model_config.num_points,
        **preprocessing,
    }
    synthetic = run["command"] == "pretrain"
    if synthetic:
        common["downsample"] = None  # generator renders at the effective input scale
    val_transform = PolylineTransform(**common)
    train_transform = PolylineTransform(
        **common,
        horizontal_flip_p=0.5,
        vertical_flip_p=0.5,
        rotation_degrees=0 if synthetic else 180,
    )
    if synthetic:
        if run["max_filaments"] > model_config.num_queries:
            raise ValueError("max_filaments must not exceed num_queries")
        options = {
            key: run[key]
            for key in SYNTHETIC_OPTIONS
            if key not in {"train_samples", "val_samples"}
        }
        options["render_size"] = run["image_size"] // preprocessing["downsample"]
        train = SyntheticFilamentDataset(
            transform=train_transform,
            num_samples=run["train_samples"],
            seed=run["seed"],
            **options,
        )
        val = SyntheticFilamentDataset(
            transform=val_transform,
            num_samples=run["val_samples"],
            seed=run["seed"] + 1_000_000,
            **options,
        )
    else:
        source = Path(run["input"])
        if source.is_dir():
            # Pair labels beside the input MRC, preserving any symlink path.
            paths = sorted(
                {
                    path.absolute()
                    for path in source.rglob("*")
                    if path.suffix.lower() == ".mrc" and path.is_file()
                }
            )
            if not paths:
                raise ValueError(f"No MRC micrographs found in {source}")
            missing = [
                path.with_suffix(".cbox")
                for path in paths
                if not path.with_suffix(".cbox").is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing sibling CBOX labels: {missing[:3]}")
            dataset = CBoxDataset(
                pd.DataFrame(
                    {
                        "image_path": paths,
                        "cbox_path": [path.with_suffix(".cbox") for path in paths],
                    }
                ),
                transform=val_transform,
            )
        elif source.suffix.lower() == ".parquet":
            dataset = ParquetDataset(source, transform=val_transform)
        else:
            raise ValueError(
                f"Expected a training directory or a Parquet file: {source}"
            )
        count = len(dataset)
        if count < 2:
            raise ValueError("An 80/20 split needs at least two micrographs")
        val_count = max(1, round(count * 0.2))
        order = torch.randperm(
            count, generator=torch.Generator().manual_seed(run["seed"])
        ).tolist()
        train_indices, val_indices = order[val_count:], order[:val_count]
        training_dataset = copy(dataset)
        training_dataset.transform = train_transform
        train, val = (
            Subset(training_dataset, train_indices),
            Subset(dataset, val_indices),
        )
    return DataModule(
        train_dataset=train,
        val_dataset=val,
        batch_size=run["batch_size"],
        num_workers=run["num_workers"],
        seed=run["seed"],
        max_steps=run.get("max_steps", 5000),
        accumulate_grad_batches=run.get("accumulate_grad_batches", 1),
    )


def run(args: argparse.Namespace):
    checkpoint_path = getattr(args, "checkpoint", None)
    checkpoint = read_checkpoint(checkpoint_path) if checkpoint_path else None
    config = resolve_run(args)
    output_dir = Path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output_dir}")
    device, precision = runtime(config["device"])
    L.seed_everything(config["seed"], workers=True)
    if checkpoint is None:
        model = PolylineModule.from_rf_detr(
            config["rf_detr_checkpoint"],
            num_queries=config["num_queries"],
            num_points=config["num_points"],
            preprocessing=config["preprocessing_config"],
            learning_rate=config["learning_rate"],
        )
    else:
        model = PolylineModule.from_checkpoint(
            checkpoint, learning_rate=config["learning_rate"]
        )
    data = build_data(config, model.model.config, model.hparams.preprocessing)
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = configure_logging(output_dir, config["command"])
    logger.info("Resolved run configuration: %s", json.dumps(config))
    (output_dir / "config.json").write_text(
        json.dumps({"run": config, "model": dict(model.hparams)}, indent=2) + "\n"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )
    callbacks = [checkpoint_callback]
    if config["early_stopping_patience"]:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=config["early_stopping_patience"],
                check_on_train_epoch_end=False,
            )
        )
    trainer = L.Trainer(
        accelerator=device,
        devices=1,
        precision=precision,
        max_steps=config["max_steps"],
        max_epochs=-1,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        val_check_interval=min(config["val_every"], config["max_steps"])
        * config["accumulate_grad_batches"],
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        default_root_dir=output_dir,
        callbacks=callbacks,
        logger=CSVLogger(output_dir, name="logs"),
        use_distributed_sampler=False,
        enable_model_summary=False,
    )
    try:
        trainer.fit(model, datamodule=data)
        logger.info("Finished at optimizer step %d", trainer.global_step)
        if checkpoint_callback.best_model_path:
            print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
