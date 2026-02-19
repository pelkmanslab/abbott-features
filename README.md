## Overview
The abbott-features Task Collection is intented to be used in combination with the [Fractal Analytics Platform](https://github.com/fractal-analytics-platform) maintained by the [BioVisionCenter Zurich](https://www.biovisioncenter.uzh.ch/en.html) (co-founded by the Friedrich Miescher Institute and the University of Zurich). 

The tasks in abbott-features are focused on extending Fractal's capabilities to extract features from (multiplexed) 3D image data.

## Available Tasks
| Task | Description | Passing |
| --- | --- | --- |
| Calculate Cycle Registration Quality | Calculates image-based registration quality across multiplexed OME-Zarr datasets.| ✓ |
| Measure Features | Calculates morphology, intensity, distance, and colocalization features for objects in a 3D label image.| ✓ |
| Get Cellvoyager Time Decay | Calculates time-decay correction factors per ROI, channel and acquisition to correct for acquisition bias dependent intensity decay (aka imaging snake).| ✓ |
| Get Z Decay Models | Calculates z-decay correction models per channel label to correct intensity decay across z.|✓|

## Installation

To install this task package on a Fractal server, get the whl in the Github release and use the local task collection.
To install this package locally:
```
git clone https://github.com/pelkmanslab/abbott-features
cd abbott
pip install -e .
```

For development:
```
git clone https://github.com/pelkmanslab/abbott-features
cd abbott
pip install -e ".[dev]" 
pre-commit install
```
to update manifest:
```
fractal-manifest create --package abbott_features
```

## Contributors
The code is based on [zfish](https://github.com/MaksHess/zfish) originally developed by [Maks Hess](https://github.com/MaksHess) and adapted to Fractal & maintained by [Ruth Hornbachner](https://github.com/rhornb).

