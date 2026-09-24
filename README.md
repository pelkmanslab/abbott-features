## Overview
The abbott-features Task Collection is intended to be used in combination with the [Fractal Analytics Platform](https://github.com/fractal-analytics-platform) maintained by the [BioVisionCenter Zurich](https://www.biovisioncenter.uzh.ch/en.html) (co-founded by the Friedrich Miescher Institute and the University of Zurich).

The tasks in abbott-features are focused on extending Fractal's capabilities to extract features from (multiplexed) 3D image data. For pre-processing of 3D multiplexed imaging data take a look at [abbott](https://github.com/pelkmanslab/abbott/tree/main). For segmentation-related tasks checkout[abbott-segmentation-tasks](https://github.com/pelkmanslab/abbott-segmentation-tasks). 

## Available Tasks
| Task | Description | Passing |
| --- | --- | --- |
| Calculate Cycle Registration Quality | Calculates image-based registration quality across multiplexed OME-Zarr datasets. | ✓ |
| Measure Features | Calculates morphology, intensity, distance, and colocalization features for objects in a 3D label image. | ✓ |
| Get Cellvoyager Time Decay | Calculates time-decay correction factors per ROI, channel and acquisition to correct for acquisition bias dependent intensity decay (aka imaging snake). | ✓ |
| Get Z Decay Models | Calculates z-decay correction models per channel label to correct intensity decay across z. | ✓ |
| Aggregate Feature Tables | Concatenates feature tables into a single table. | ✓ |

## Installation

### On a Fractal server

Download the `.tar.gz` from the latest [GitHub release](https://github.com/pelkmanslab/abbott-features/releases) and install it with Fractal's pixi task collection.

### Locally

Requires Python 3.11–3.13.

```bash
git clone https://github.com/pelkmanslab/abbott-features
cd abbott-features
pip install -e .
```

## Development

The development environment is managed with [pixi](https://pixi.sh):

```bash
git clone https://github.com/pelkmanslab/abbott-features.git
cd abbott-features
pixi run init-tasks    # install pre-commit hooks, format code, build manifest, run tests
```

Individual tasks:

```bash
pixi run -e dev create-manifest   # regenerate __FRACTAL_MANIFEST__.json
pixi run -e dev format-code       # ruff format
pixi run -e test test             # run the test suite
```

## Contributors
The code is based on [zfish](https://github.com/MaksHess/zfish) originally developed by [Maks Hess](https://github.com/MaksHess) and adapted to Fractal & maintained by [Ruth Hornbachner](https://github.com/rhornb).
