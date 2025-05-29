### Purpose
- Calculates z-decay correction models per channel label to correct intensity decay across z.
- Z decay models can be of type Exp, Linear and LogLinear calculated with linear or huber loss. For each type, z decay bias is corrected either 1D or 2D (by partioning light path into medium and sample path).
- Z decay models are available in three types: Exponential (Exp), Linear, and LogLinear, each computed using either linear or Huber loss. For all model types, z-decay bias correction models are saved as 1D (uniform correction across the entire light path) and 2D mode (with light path divided into medium path and sample path).

### Outputs
- Z-decay models saved in the OME-Zarr plate /path_to_ome_zarr_fld/models/z_decay/ 
- Plots saved in /path_to_ome_zarr_fld/models/__plots containing overview of 
1. `overview__one_step` / `overview__two_step` - 1D & 2D decay models fit to channels.
2. `equivalent_spherical_radius_cutoff` - adjustable via the `spherical_radius_cutoff` parameter used for outlier removal.
3. `roundness_cutoff` - adjustable via the `roundness_cutoff` parameter used for outlier removal.
