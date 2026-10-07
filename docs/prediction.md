# Prediction

Start with a local or Hugging Face CryoTracer checkpoint and MRC micrographs.
Quote the input glob so CryoTracer receives the pattern and expands it itself.
MRC headers must contain a positive pixel size.

## Export CBOX picks

Replace `provided.ckpt` with your checkpoint path:

```sh
cryotracer predict 'MotionCorr/job003/movies1/*.mrc' \
  --checkpoint provided.ckpt \
  --box-size 512 --output-dir predictions
```

Run from your RELION project directory and adjust the path as needed.
CryoTracer automatically skips files ending exactly in `_PS.mrc`, even when
the glob matches them.

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

## Convert saved picks for RELION

Prediction always saves CBOX filaments. Use the separate
[`cryotracer relion` command](relion.md) to sample the saved filaments into
RELION coordinate STAR files. Sampling parameters can be changed without
running prediction again.

Run `cryotracer predict --help` for all options.
