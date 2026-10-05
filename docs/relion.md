# RELION conversion

Convert saved [CBOX picks](prediction.md) into particle coordinates without
running prediction again:

```sh
cryotracer relion predictions/cbox/ \
  --micrographs-star CtfFind/job003/micrographs_ctf.star \
  --box-size 512 --output-dir predictions-relion \
  --helical-rise-angstrom 4.75 --helical-asym-units 10
```

Use a micrographs STAR with optics metadata and filenames matching the CBOX
files. The MRC images must be available, and their pixel sizes must agree with
the STAR.

Particle spacing is **rise × asymmetric units**: 47.5 Å in this example.
Choose these values for your specimen. `--box-size` is the extraction box size
in original image pixels and must be even.

The command writes one coordinate STAR per micrograph and a
`predictions-relion/coordinates.star` list. Filament IDs and cumulative track
distances are preserved through bends and gaps.

## Extract in RELION 5

Create an **Extract** job with these inputs and settings:

| Setting                          | Value                                 |
| -------------------------------- | ------------------------------------- |
| Micrograph STAR                  | Your existing CTF micrographs STAR    |
| Input coordinates                | `predictions-relion/coordinates.star` |
| Re-extract refined particles?    | No                                    |
| Extract helical segments?        | Yes                                   |
| Coordinates are start-end only?  | No                                    |
| Particle box size                | Same as the conversion's `--box-size` |
| Tube diameter and angular priors | Appropriate values for your specimen  |

Run Extract in the same RELION project as the micrographs STAR. RELION extracts
the images and adds CTF and optics metadata. Use the resulting
`Extract/jobXXX/particles.star` for classification and helical reconstruction.

To change particle spacing, rerun conversion with new parameters and a new
output directory, or add `--overwrite`. The CBOX files stay unchanged.

## Crossing removal

By default, samples within 140 Å along either side of a crossing are removed.
Adjust this with `--crossing-exclusion-angstrom`, or disable it with
`--no-crossing-removal`. Boxes extending beyond the image are also skipped.

Removal preserves filament IDs and track distances, leaving gaps that RELION
can use. Choose the exclusion distance for your box size and crossing geometry;
retained boxes can still contain crossing signal.

See `cryotracer relion --help` for other options.
