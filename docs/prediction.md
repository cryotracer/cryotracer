# Prediction

Start with a local or Hugging Face CryoTracer checkpoint and MRC micrographs.
Quote the input glob so CryoTracer receives the pattern and expands it itself.
MRC headers must contain a positive pixel size.

## Export CBOX picks

Replace `provided.ckpt` with your checkpoint path:

```sh
cryotracer predict 'data/**/*.mrc' \
  --checkpoint provided.ckpt \
  --box-size 512 --output-dir predictions
```

You can also use `--checkpoint hf://owner/repo/file.ckpt`; CryoTracer downloads
and caches it automatically.

`--box-size` is the box width in original image pixels. CryoTracer writes one
file per input image under `predictions/cbox/`, including an empty CBOX file
when no filaments pass the prediction threshold:

```text
predictions/
  cbox/
    image_001.cbox
    image_002.cbox
  predict.log
```

Input filenames must have unique stems for CBOX export: two images named
`image_001.mrc` in different folders would map to the same output file.

## Adjust predictions

By default, prediction uses the score threshold stored in the checkpoint.
Override it with `--prediction-threshold`, using a value between 0 and 1:

```sh
cryotracer predict 'data/**/*.mrc' \
  --checkpoint provided.ckpt \
  --box-size 160 --output-dir predictions-threshold \
  --prediction-threshold 0.5
```

Existing prediction files cause the command to stop. Use a different output
directory or add `--overwrite` when you intend to replace those files.

## Convert saved picks for RELION

Prediction always saves CBOX filaments. Use the separate
[`cryotracer relion` command](relion.md) to sample the saved filaments into
RELION coordinate STAR files. Sampling parameters can be changed without
running prediction again.

Run `cryotracer predict --help` for all options.
