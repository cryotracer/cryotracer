# CryoTracer

CryoTracer finds filaments in cryo-EM micrographs and saves the picks as CBOX
files. Convert the saved filaments to RELION coordinates in a separate step.
Start with a pretrained CryoTracer checkpoint and your MRC images.

## Install

Use Python 3.11–3.14 and Git. Create a virtual environment and install
CryoTracer directly from GitHub:

```sh
python3 -m venv ~/cryotracer-venv
source ~/cryotracer-venv/bin/activate
python -m pip install "git+https://github.com/cryotracer/cryotracer.git"
cryotracer --help
```

Activate the same environment when you return to use CryoTracer.

## Pretrained checkpoint

A pretrained checkpoint for alpha-synuclein filament picking is available on
[Hugging Face](https://huggingface.co/cryotracer/cryotracer-alpha-synuclein).

## Predict filaments

Pick filaments with the pretrained alpha-synuclein checkpoint:

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

`--box-size` sets the CBOX box width in the original image pixels. CryoTracer
writes one CBOX file per MRC image in the `cbox/` subdirectory, such as
`predictions/cbox/image_001.cbox`.
It also writes a CBOX file when it finds no filaments.

### Convert CBOX picks for RELION

Convert saved CBOX files with the matching RELION micrographs STAR and your
helical sampling parameters. You can change the spacing without running
prediction again:

```sh
cryotracer relion predictions/cbox/ \
  --micrographs-star CtfFind/job003/micrographs_ctf.star \
  --box-size 512 --output-dir predictions-relion \
  --helical-rise-angstrom 4.75 --helical-asym-units 10
```

This writes one coordinate STAR per micrograph in `predictions-relion/coords/`
and a `predictions-relion/coordinates.star` list for RELION's Extract job.
Particle spacing is **rise × asymmetric units**: 47.5 Å in this example.

In RELION Extract, select your CTF micrographs STAR and the exported coordinate
list. Set **Extract helical segments?** to **Yes** and **Coordinates are start-end
only?** to **No**. RELION extracts the images and attaches CTF and optics metadata.
See the [RELION conversion guide](docs/relion.md) for details.

## Fine-tune your checkpoint

If you have new labelled images, place each MRC beside its matching CBOX
file:

```text
labelled/
  image_001.mrc
  image_001.cbox
  image_002.mrc
  image_002.cbox
```

Then fine-tune the pretrained alpha-synuclein checkpoint:

```sh
cryotracer train labelled/ \
  --checkpoint hf://cryotracer/cryotracer-alpha-synuclein/best.ckpt \
  --output-dir runs/finetuned
```

CryoTracer searches the input directory recursively for MRC/CBOX pairs.

Use at least two image pairs and a new or empty output directory. The fine-tuned
checkpoint is saved at `runs/finetuned/checkpoints/best.ckpt`. Use that
path with `predict` when you want to pick with the fine-tuned model.

Run `cryotracer predict --help`, `cryotracer relion --help`, or
`cryotracer train --help` for more options.

## EMPIAR-10022 dataset

The [EMPIAR-10022 filament dataset](https://huggingface.co/datasets/cryotracer/empiar10022-filaments)
contains 109 cryo-EM micrographs with matching CBOX filament annotations.

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

## License

CryoTracer's original code and modifications are licensed under
[GNU GPL version 3 only](LICENSE). Portions adapted from Hugging Face
Transformers, RF-DETR, and Deformable DETR retain their Apache-2.0 terms and
copyright notices. The full text of both licenses is included in `LICENSE`.
