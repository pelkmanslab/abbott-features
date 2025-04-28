"""This is the Python module for my_task."""

import logging
from typing import Optional

import numpy as np
from ngio import open_ome_zarr_container
from pydantic import validate_call
from skimage.measure import label
from skimage.morphology import ball, dilation, opening, remove_small_objects


@validate_call
def thresholding_label_task(
    *,
    zarr_url: str,
    threshold: int,
    channel: str,
    label_name: Optional[str] = None,
    min_size: int = 50,
    overwrite: bool = True,
) -> None:
    """Threshold an image and find connected components.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
        threshold: Threshold value to be applied.
        channel: Channel label to be thresholded.
        label_name: Name of the resulting label image
        min_size: Minimum size of objects. Smaller objects are filtered out.
        overwrite: Whether to overwrite an existing label image
    """
    # Use the first of input_paths
    logging.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)

    logging.info(f"{ome_zarr=}")

    image = ome_zarr.get_image()
    logging.info(f"{image=}")

    if channel not in image.channel_labels:
        raise ValueError(
            f"Channel {channel} not found in image channels: {image.channel_labels}"
        )

    channel_index = image.channel_labels.index(channel)
    logging.info(f"channel {channel} found at index: {channel_index}")

    # Load the highest-resolution multiscale array through dask.array
    array = image.get_array(c=channel_index, mode="numpy")

    # Process the image with an image processing approach of your choice
    label_array = process_img(
        array,
        threshold=threshold,
        min_size=min_size,
    )
    
    # Set label name
    if not label_name:
        label_name = f"{channel}_thresholded"

    label_image = ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    logging.info(f"Label image {label_name} created")
    logging.info(f"{label_image=}")

    label_image.set_array(patch=label_array)
    label_image.consolidate()
    logging.info("Label image set and consolidated")


def process_img(int_img: np.ndarray, threshold: int, min_size: int = 50) -> np.ndarray:
    """Image processing function, to be replaced with your custom logic

    Numpy image & parameters in, label image out

    Args:
        int_img (np.ndarray): Input image to be processed
        threshold (int): Threshold value for binarization
        min_size (int): Minimum size of objects to keep

    Returns:
        label_img (np.ndarray): Labeled image
    """
    # Thresholding the image
    int_img = np.squeeze(int_img)
    
    binary_img = int_img >= threshold

    # Removing small objects
    cleaned_img = remove_small_objects(binary_img, min_size=min_size)
    # Opening to separate touching objects
    selem = ball(1)
    opened_img = opening(cleaned_img, selem)

    # Optional: Dilation to restore object size
    dilated_img = dilation(opened_img, selem)

    # Labeling the processed image
    label_img = label(dilated_img, connectivity=1)

    return label_img


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=thresholding_label_task)