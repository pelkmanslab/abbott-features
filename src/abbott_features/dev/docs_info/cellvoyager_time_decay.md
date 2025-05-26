### Purpose
- Calculates time-decay models per ROI, channel and acquisition to correct for acquisition bias dependent intensity decay (imaging snake).
- Time decay models can be of type Exp, ExpNoOffset, Linear and LogLinear calculated with linear loss.
- 
- Supports **2D (not tested) and 3D measurements** across multiple regions of interest (ROIs).

### Outputs
- A  **generic table** saved in the OME-Zarr plate /path_to_ome_zarr_fld/tables with .parquet backend.
- Two plots saved in /path_to_ome_zarr_fld/__plots : 1. equivalent_spherical_radius_cutoff (can be modified by tuning `spherical_radius_cutoff` parameter) and 2. time_decay_models containing all decay models per channel name.
