# Fine-tuning

Fine-tune a provided CryoTracer checkpoint on labelled micrographs from your
dataset. Training loads the model and preprocessing settings from the checkpoint.

## Prepare labelled images

Place each MRC image beside a CBOX label file with the same stem:

```text
labelled/
  image_001.mrc
  image_001.cbox
  image_002.mrc
  image_002.cbox
```

CryoTracer searches the input directory recursively for MRC images. Each image
must have its sibling CBOX file. You can also pass a Parquet metadata file
directly, such as `labelled/metadata.parquet`. Use at least two micrographs.
CryoTracer makes an 80/20 training and validation split, with at least one
validation image, using `--seed` for reproducibility.

## Train from a checkpoint

Use a new or empty output directory:

```sh
cryotracer train labelled/ \
  --checkpoint provided.ckpt \
  --output-dir runs/finetuned
```

You can also use `--checkpoint hf://owner/repo/file.ckpt`; CryoTracer downloads
and caches it automatically.

The checkpoint determines the model's query count, polyline point count, and
preprocessing. Do not supply `--num-queries` or `--num-points` when loading a
checkpoint. Training controls can still be changed.

Use `--batch-size` to control how many micrographs are processed together. The
default is 8; reduce it if you run out of memory. `--num-workers` defaults to 4,
and setting it to 0 loads data in the main process.

The best checkpoint is selected by the lowest validation loss and saved to
`runs/finetuned/checkpoints/best.ckpt`.

## Pick with the fine-tuned model

Pass the new checkpoint to prediction:

```sh
cryotracer predict 'MotionCorr/job003/movies1/*fractions.mrc' \
  --checkpoint runs/finetuned/checkpoints/best.ckpt \
  --box-size 512 --output-dir predictions-finetuned
```

Adjust the directory for your project and keep the input glob quoted. Select
only micrograph files with `*fractions.mrc` for RELION, or
`*patch_aligned_doseweighted.mrc` or `*patch_aligned.mrc` for CryoSPARC.

See the [prediction guide](prediction.md) for CBOX and RELION export. Run
`cryotracer train --help` for all training options.
