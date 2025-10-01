"""Init registration module for image-based registration based on tasks-core."""

import logging
from typing import Any, Optional

from fractal_tasks_core.tasks.image_based_registration_hcs_init import (
    image_based_registration_hcs_init,
)
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def init_registration_quality_hcs(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    reference_acquisition: int = 0,
    zarr_ending: Optional[str] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Initializes registration quality control for HCS OME-Zarrs.

    This task prepares a parallelization list of all zarr_urls that need to be
    used to calculate the registration between acquisitions (all zarr_urls
    except the reference acquisition vs. the reference acquisition).
    This task only works for HCS OME-Zarrs for 2 reasons: Only HCS OME-Zarrs
    currently have defined acquisition metadata to determine reference
    acquisitions. And we have only implemented the grouping of images for
    HCS OME-Zarrs by well (with the assumption that every well just has 1
    image per acqusition).

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_acquisition: Which acquisition was used as reference. Needs to
            match the acquisition metadata in the OME-Zarr image.
        zarr_ending: Optional; file ending of the OME-Zarrs if multiple registered
            OME-Zarrs are present in the same directory. Default: None.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    if zarr_ending is not None:
        zarr_urls = [z for z in zarr_urls if z.endswith(zarr_ending)]

    return image_based_registration_hcs_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        reference_acquisition=reference_acquisition,
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=init_registration_quality_hcs,
        logger_name=logger.name,
    )
