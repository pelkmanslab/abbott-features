"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    CompoundTask,
    NonParallelTask,
    ParallelTask,
)

AUTHORS = "Ruth Hornbachner, Maks Hess"
DOCS_LINK = "https://github.com/pelkmanslab/abbott-features"

TASK_LIST = [
    ParallelTask(
        name="Measure Features",
        executable="fractal_tasks/measure_features.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Measurement",
        tags=[
            "3D",
            "Morphology",
            "Intensity",
            "Distance",
            "Colocalization",
            "regionprops",
            "itk",
            "Feature Table",
        ],
        docs_info="file:docs_info/measure_features.md",
    ),
    NonParallelTask(
        name="Get Cellvoyager Time Decay",
        executable="fractal_tasks/cellvoyager_time_decay.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Measurement",
        tags=[
            "3D",
            "Yokogawa",
            "Intensity Correction",
            "Feature Table",
        ],
        modality="HCS",
        docs_info="file:docs_info/cellvoyager_time_decay.md",
    ),
    NonParallelTask(
        name="Get Z Decay Models",
        executable="fractal_tasks/z_decay.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Measurement",
        tags=[
            "3D",
            "Intensity Correction",
            "Feature Table",
        ],
        modality="HCS",
        docs_info="file:docs_info/z_decay.md",
    ),
    CompoundTask(
        name="Calculate Cycle Registration Quality",
        executable_init="fractal_tasks/init_registration_quality_hcs.py",
        executable="fractal_tasks/cycle_registration_quality.py",
        input_types=dict(registered=True),
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Registration",
        modality="HCS",
        tags=["Multiplexing", "3D"],
        docs_info="file:docs_info/calculate_registration_quality.md",
    ),
]
