# Fine-tuning

Fine-tune a provided CryoTracer checkpoint on labelled micrographs from your
dataset. Training loads the model and preprocessing settings from the checkpoint.

## Prepare annotations with napari

Use [napari](https://napari.org/) with the
[napari-boxmanager plugin](https://github.com/MPI-Dortmund/napari-boxmanager)
to trace filaments manually. Follow the plugin's installation
instructions in a separate environment from CryoTracer, and run the GUI on a
workstation with a graphical display.

Choose at least two representative micrographs from the dataset you want to
pick, including variation in filament density, contrast, and background. Copy
the selected MRC files into `labelled/`, keeping their original dimensions and
pixel size. Open them in the annotation environment:

```sh
napari_boxmanager 'labelled/*.mrc'
```

Keep the input glob quoted.

!!! warning "Micrographs must be fully annotated"
    Fully annotate every micrograph used for training or validation: trace every
    visible target filament across the entire image, over its full visible
    length. Annotating only a few filaments or a convenient region is not enough.
    Unannotated filaments can be treated as background during training, teaching
    the model to suppress real filaments. Prefer fewer completely annotated
    micrographs to more partially annotated ones. Exclude unfinished micrographs
    until their annotations are complete.

1. Create a filament layer in napari-boxmanager. Trace each filament's
   centreline as a separate path with at least two ordered vertices. Add enough
   vertices to follow curves accurately, and keep crossing filaments as separate
   paths.
2. Inspect the whole image at a useful zoom level. Add every missed filament,
   correct paths and endpoints, and remove false positives and duplicate traces.
3. Export the reviewed filament layer as **CBOX** using napari-boxmanager's
   `organize_layer` export controls.
4. Save each CBOX file beside its MRC image with the same stem.
   Reopen the saved annotations with the images to check that the paths align
   and the complete set of filaments was saved.

After saving, your `labelled/` directory should contain matching image and
annotation files:

```text
labelled/
  image_001.mrc
  image_001.cbox
  image_002.mrc
  image_002.cbox
```

CryoTracer searches the input directory recursively for MRC images. Each image
must have its sibling CBOX file.

## Train from a checkpoint

Pass the completed `labelled/` directory to training with a provided checkpoint.
Use a new or empty output directory:

```sh
cryotracer train labelled/ \
  --checkpoint provided.ckpt \
  --output-dir runs/finetuned
```

CryoTracer makes an 80/20 training and validation split, with at least one
validation image, using `--seed` for reproducibility.

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
cryotracer predict 'MotionCorr/job002/movies1/*fractions.mrc' \
  --checkpoint runs/finetuned/checkpoints/best.ckpt \
  --box-size 512 --output-dir predictions-finetuned
```

Adjust the directory for your project and keep the input glob quoted. Select
only micrograph files with `*fractions.mrc` for RELION, or
`*patch_aligned_doseweighted.mrc` or `*patch_aligned.mrc` for CryoSPARC.

See the [prediction guide](prediction.md) for CBOX and RELION export. Run
`cryotracer train --help` for all training options.
