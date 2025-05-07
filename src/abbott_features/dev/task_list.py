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
    ParallelTask(
        name="Measure Intensity Features",
        executable="fractal_tasks/measure_intensity_features.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["3D", "Morphology", "Feature Table"],
        docs_info="file:docs_info/measure_intensity_features.md",
    ),
    ParallelTask(
        name="Measure Colocalization Features",
        executable="fractal_tasks/measure_colocalization_features.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["3D", "Morphology", "Feature Table"],
        docs_info="file:docs_info/measure_colocalization_features.md",
    ),
    ParallelTask(
        name="Measure Distance Features",
        executable="fractal_tasks/measure_distance_features.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=["3D", "Morphology", "Feature Table"],
        docs_info="file:docs_info/measure_distance_features.md",
    ),
]
