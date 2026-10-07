# CryoTracer

CryoTracer picks filaments in cryo-EM micrographs using a local or Hugging Face
model checkpoint. Prediction saves filament picks as CBOX files. A separate
conversion command samples the saved filaments for RELION extraction.

## Start picking

You need MRC micrographs and a CryoTracer checkpoint. The checkpoint contains
the model and preprocessing settings used for prediction.

1. [Install CryoTracer](installation.md) and check that the command is available.
1. [Predict filaments](prediction.md) to save CBOX picks.
1. [Convert CBOX picks to RELION coordinates](relion.md) for particle extraction.
1. [Fine-tune a checkpoint](fine-tuning.md) if you have labelled micrographs for
   your dataset.

The prediction guide explains box sizes and output files.
The RELION guide covers conversion and extraction. The fine-tuning guide covers
MRC/CBOX pairs, training controls, and using the resulting checkpoint.

## Command help

Each command lists its options and defaults:

```sh
cryotracer --help
cryotracer predict --help
cryotracer relion --help
cryotracer train --help
```
