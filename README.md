# Overview
The abbott-features Task Collection is intented to be used in combination with the [Fractal Analytics Platform](https://github.com/fractal-analytics-platform) maintained by the [BioVisionCenter Zurich](https://www.biovisioncenter.uzh.ch/en.html) (co-founded by the Friedrich Miescher Institute and the University of Zurich). 

The tasks in abbott-features are focused on extending Fractal's capabilities to extract features from 3D image data using [Polars](https://github.com/pola-rs/polars) and .parquet backend.

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
fractal-manifest create --package abbott_features --fractal-server-2-13
```

# Contributors
The code is based on [zfish](https://github.com/MaksHess/zfish) originally developed by [Maks Hess](https://github.com/MaksHess) and adapted to Fractal & maintained by [Ruth Hornbachner](https://github.com/rhornb).

