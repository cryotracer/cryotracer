import logging

import torch
from tqdm.auto import tqdm

from cryotracer.data.cbox import write_cbox
from cryotracer.data.datamodule import DataModule
from cryotracer.data.prediction import PredictionDataset, expand_micrographs
from cryotracer.data.transforms import PolylineTransform
from cryotracer.model.module import PolylineModule, read_checkpoint, resolve_checkpoint
from cryotracer.train import configure_logging, runtime

logger = logging.getLogger(__name__)


def run(args):
    paths = expand_micrographs(args.input)
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Input filenames must have unique stems for CBOX export")
    output_dir = args.output_dir.expanduser().resolve()
    cbox_dir = output_dir / "cbox"
    output_paths = [cbox_dir / f"{path.stem}.cbox" for path in paths]
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    inputs = {checkpoint_path.resolve(), *paths}
    for path in output_paths:
        if path.resolve() in inputs:
            raise ValueError(f"Output would overwrite an input: {path}")
        if path.is_dir():
            raise IsADirectoryError(path)
        if (path.exists() or path.is_symlink()) and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite")
    checkpoint = read_checkpoint(checkpoint_path)
    model = PolylineModule.from_checkpoint(checkpoint)
    dataset = PredictionDataset(
        paths,
        PolylineTransform(
            num_queries=model.model.config.num_queries,
            num_points=model.model.config.num_points,
            **model.hparams.preprocessing,
        ),
    )
    data = DataModule(
        predict_dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device, precision = runtime(args.device)
    model.to(device).eval()
    threshold = (
        model.hparams.prediction_threshold
        if args.prediction_threshold is None
        else args.prediction_threshold
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = configure_logging(output_dir, "predict")
    logger.info(
        "Predicting %d micrographs from %s, threshold=%s",
        len(paths),
        args.checkpoint,
        threshold,
    )
    try:
        with (
            torch.inference_mode(),
            tqdm(total=len(paths), desc="Predicting", unit="micrograph") as progress,
        ):
            for batch in data.predict_dataloader():
                with torch.autocast(
                    device_type=device,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16-mixed",
                ):
                    outputs = model(batch["pixel_values"].to(device))
                scores = 1 - outputs.logits.float().softmax(-1)[..., -1]
                points = outputs.pred_polylines.float().cpu()
                for index, image_id in enumerate(batch["image_id"].tolist()):
                    keep = scores[index].cpu() >= threshold
                    polylines = (
                        points[index, keep] * batch["native_size"][index]
                    ).tolist()
                    write_cbox(
                        output_paths[image_id], polylines, box_size=args.box_size
                    )
                    logger.info(
                        "Wrote %s (%d filaments)",
                        output_paths[image_id],
                        len(polylines),
                    )
                    progress.update(1)
        print(f"Saved {len(paths)} CBOX files to {cbox_dir}")
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
