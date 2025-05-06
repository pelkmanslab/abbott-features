"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Ruth Hornbachner, Maks Hess"
DOCS_LINK = "https://github.com/pelkmanslab/abbott-features"

TASK_LIST = [
    ParallelTask(
        name="Measure Label Features",
        executable="fractal_tasks/measure_label_features.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["3D", "Morphology", "Feature Table"],
        docs_info="file:docs_info/measure_label_features.md",
    ),
]
