# from collections import namedtuple

# Patient = namedtuple("Patient", ["geometry", "data"])

import warnings as _warnings

from pulse import annotation
from pulse.utils import Text, make_logger

from . import (
    assimilator,
    model_observations,
    observations,
    optimal_control,
    optimization_targets,
    regularization,
)
from .assimilator import Assimilator
from .model_observations import (
    BoundaryObservation,
    ModelObservation,
    StrainObservation,
    VolumeObservation,
)
from .observations import Observation, Observations
from .optimal_control import OptimalControl
from .optimization_targets import OptimizationTarget
from .reduced_functional import ReducedFunctional
from .regularization import Regularization

annotation.annotate = True

_warnings.filterwarnings("ignore", category=UserWarning)

__version__ = "2.0"
__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"

__all__ = [
    "assimilator",
    "model_observations",
    "observations",
    "optimal_control",
    "optimization_targets",
    "regularization",
    "Assimilator",
    "BoundaryObservation",
    "ModelObservation",
    "StrainObservation",
    "VolumeObservation",
    "Observation",
    "Observations",
    "OptimalControl",
    "OptimizationTarget",
    "ReducedFunctional",
    "Regularization",
    "make_logger",
    "Text",
    "annotation",
]
