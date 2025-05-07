### Purpose
- Calculates **distance features** for objects in a 3D label image relative to a label mask e.g. nuclei to embryo mask.
- Supports **3D measurements** across multiple regions of interest (ROIs).

### Outputs
- A **distance feature table** saved in the OME-Zarr structure, containing:
- Distance features (e.g., distance to border, distance along axes, etc.)
