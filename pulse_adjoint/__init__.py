import warnings as _warnings

_warnings.filterwarnings("ignore", category=UserWarning)

# from .setup_parameters import setup_adjoint_contraction_parameters
# parameters = setup_adjoint_contraction_parameters()

from collections import namedtuple

Patient = namedtuple("Patient", ["geometry", "data"])
from pulse import annotation
annotation.annotate = True
from pulse.utils import make_logger, Text

from . import dolfin_utils
from .dolfin_utils import ReducedFunctional


from . import config

from . import optimal_control
from .optimal_control import OptimalControl

from . import assimilator
from .assimilator import Assimilator

from . import regularization
from .regularization import Regularization

from . import optimization_targets
from .optimization_targets import OptimizationTarget

from .reduced_functional import ReducedFunctional

from . import model_observations
from .model_observations import (ModelObservation,
                                 BoundaryObservation,
                                 VolumeObservation,
                                 StrainObservation)


__version__ = "2.0"
__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
