### Purpose
- **Calculates image-based registration quality** across multiplexed OME-Zarr datasets.
- Useful to automatically identify misaligned embryos / organoids across cycles to realign / exclude from downstream analysis.
- Requires embryo / organoid segmentation and corresponding masking ROI table.
- Calculates image-based Correlation Scores (Pearson, Spearman, Kendall) for each ROI across cycles.
- Typically used following image-based registration tasks such as `Compute/Apply Registration (elastix)` or `Compute/Apply Registration (warpfield)`.

### Output
- Dataframe containing Correlation Score between moving and reference acquisition for each embryo/organoid.

