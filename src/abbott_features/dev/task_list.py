"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Fractal Core Team"
DOCS_LINK = None
INPUT_MODELS = []

TASK_LIST = [
    ParallelTask(
        name="Thresholding Label Task",
        executable="thresholding_label_task.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation"],
        docs_info="file:docs_info/thresholding_task.md",
    ),
]
