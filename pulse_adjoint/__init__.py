import warnings as _warnings

_warnings.filterwarnings("ignore", category=UserWarning)

# from .setup_parameters import setup_adjoint_contraction_parameters
# parameters = setup_adjoint_contraction_parameters()

from collections import namedtuple

Patient = namedtuple("Patient", ["geometry", "data"])
from pulse import annotation
from pulse.utils import make_logger, Text

from . import dolfin_utils
from .dolfin_utils import ReducedFunctional

# from . import forward_runner
# from . import setup_optimization
# from . import run_optimization

# from . import utils
# from . import dolfin_utils
# from . import io_utils
from . import config

# from . import heart_problem
from . import optimal_control
# from . import store_results

# from . import clinical_data
# from .clinical_data import ClinicalData

from . import assimilator
from .assimilator import Assimilator

from . import optimization_targets
from .optimization_targets import OptimizationTarget

from .model_observations import (ModelObservation,
                                 BoundaryObservation,
                                 VolumeObservation,
                                 StrainObservation)


# Subpackages
# from . import postprocess
# from . import unloading


# from .utils import logger
# from .dolfin_utils import RegionalParameter


__version__ = "2.0"
__author__ = "Henrik Finsberg"
__credits__ = ["Henrik Finsberg"]
__license__ = "LGPL-3"
__maintainer__ = "Henrik Finsberg"
__email__ = "henriknf@simula.no"
