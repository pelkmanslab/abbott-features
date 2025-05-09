### Purpose
- Calculates **morphology**, **intensity**, **distance**, and **colocalization features** for objects in a 3D label image.
- Supports **2D (not tested) and 3D measurements** across multiple regions of interest (ROIs).

### Outputs
- A  **feature table** saved in the OME-Zarr structure with .parquet backend.
  - Morphology features 
  - Intensity features (e.g., mean, max, min intensity per object).
  - Distance features (e.g., densities and number of neighbours).
- Updated ROI metadata with border and well location information.

### Limitations
- Does not support measurements for label images that do not have the same resolution as the intensity images.