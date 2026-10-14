# CryoTracer

CryoTracer picks filaments in cryo-EM micrographs using a local or Hugging Face
model checkpoint. Prediction saves filament picks as CBOX files. A separate
conversion command samples the saved filaments for RELION extraction.

## Start picking

You need MRC micrographs and a CryoTracer checkpoint. The checkpoint contains
the model and preprocessing settings used for prediction.

The [pretrained alpha-synuclein checkpoint](https://huggingface.co/cryotracer/cryotracer-alpha-synuclein)
is available on Hugging Face for prediction and fine-tuning. The guides below
show how to download and cache it automatically with `--checkpoint`.

1. [Install CryoTracer](installation.md) and check that the command is available.
2. [Predict filaments](prediction.md) to save CBOX picks.
3. [Convert CBOX picks to RELION coordinates](relion.md) for particle extraction.
4. [Fine-tune a checkpoint](fine-tuning.md) if you have labelled micrographs for
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

## Citation

If you use CryoTracer in your research, please cite
[our paper](https://arxiv.org/abs/2610.18696):

```bibtex
@misc{kuczbanski2026cryotracer,
  title = {{CryoTracer}: A Synthetic-to-Real System for Filament Instance Tracing in {Cryo-EM}},
  author = {Kuczbański, Radosław and Kozłowski, Wojciech and
            Grzesiuk, Mateusz and Karanowski, Konrad and Frey, Lukas and
            Rzepiela, Andrzej and Zięba, Maciej},
  year = {2026},
  eprint = {2610.18696},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2610.18696}
}
```
