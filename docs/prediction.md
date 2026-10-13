# Prediction

Start with a local or Hugging Face CryoTracer checkpoint and MRC micrographs.
Quote the input glob so CryoTracer receives the pattern and expands it itself.
MRC headers must contain a positive pixel size.

## Export CBOX picks

Use the [pretrained alpha-synuclein checkpoint](https://huggingface.co/cryotracer/cryotracer-alpha-synuclein)
to pick filaments:

```sh
cryotracer predict 'MotionCorr/job002/movies1/*fractions.mrc' \
  --checkpoint hf://cryotracer/cryotracer-alpha-synuclein/best.ckpt \
  --box-size 512 --output-dir predictions
```

For RELION or CryoSPARC data, use a specific filename pattern to select only
micrograph files:

- RELION: `*fractions.mrc`
- CryoSPARC: `*patch_aligned_doseweighted.mrc` or `*patch_aligned.mrc`

Adjust the directory for your project and keep the input glob quoted.

CryoTracer downloads and caches the Hugging Face checkpoint automatically.
To use a local checkpoint, pass its path instead, such as
`--checkpoint provided.ckpt`.

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
