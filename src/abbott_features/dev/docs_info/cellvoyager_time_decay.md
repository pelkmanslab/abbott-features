### Purpose
- Calculates time-decay correction factors per ROI, channel and acquisition to correct for acquisition bias dependent intensity decay (aka imaging snake).
- Time decay models can be of type Exp, ExpNoOffset, Linear and LogLinear calculated with linear loss.
- Supports **2D (not tested) and 3D measurements** across multiple regions of interest (ROIs).

### Outputs
- A  **generic table** saved in the OME-Zarr plate /path_to_ome_zarr_fld/tables with .parquet backend.
- Two plots saved in /path_to_ome_zarr_fld/__plots : 
1. `equivalent_spherical_radius_cutoff` - adjustable via the `spherical_radius_cutoff` parameter used for outlier removal.
2. `time_decay_models` - fitted time decay correction models for each channel label

### Limitations
- This task only supports time-decay correction for images from CellVoyager microscopes.