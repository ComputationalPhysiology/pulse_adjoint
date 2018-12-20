import numpy as np
import dolfin
import dolfin_adjoint
from pulse import numpy_mpi

from .model_observations import ModelObservation
from . import make_logger, annotation


logger = make_logger(__name__, 10)


class OptimizationTarget(object):
    """
    FIXME
    """

    def __init__(self, observations, model_observation, weight=1.0, collect=True):

        msg = "Expected"
        assert isinstance(model_observation, ModelObservation)
        self.model = model_observation

        if np.isscalar(observations):
            self.observations = (observations,)
        elif isinstance(observations, tuple):
            self.observations = observations
        else:
            self.observations = tuple(observations)

        self._V = dolfin.FunctionSpace(model_observation.mesh, "R", 0)
        self._functional = dolfin_adjoint.Function(
            self._V, name="functional {}".format(self.model)
        )

        self.model_function = dolfin_adjoint.Function(
            self.model._V, name="model function {}".format(self.model)
        )
        self.data_function = dolfin_adjoint.Function(
            self.model._V, name="data function {}".format(self.model)
        )

        # Test and trial functions for the target space
        self._trial = dolfin.TrialFunction(self._V)
        self._test = dolfin.TestFunction(self._V)

        self.weight = dolfin_adjoint.Constant(weight)

        self._count = -1
        self.__len__ = len(self.observations)
        self._load_observations()

        self.collector = dict(model=[], data=[], functional=[])
        self.collect = collect

    def __repr__(self):
        return (
            "{self.__class__.__name__}" "({self.observations}, {self.model})"
        ).format(self=self)

    def _load_observations(self):
        """
        Load the observations into dolfin functions so that dolfin-adjoint
        is able to record it.
        """
        # We do not want to record this step
        annotate = annotation.annotate
        annotation.annotate = False

        dolfin_observations = []
        for obs in self.observations:
            f = dolfin.Function(self.model._V)
            if np.isscalar(obs):
                obs_arr = np.array([obs])
            else:
                obs_arr = np.array(obs)
            numpy_mpi.assign_to_vector(f.vector(), obs_arr)
            dolfin_observations.append(f)

        self.dolfin_observations = tuple(dolfin_observations)
        annotation.annotate = annotate

    def form(self):
        """
        The form of the mismatch between data and model.
        """
        return (self.data_function - self.model_function) ** 2

    @property
    def functional(self):
        return self.weight * self._functional

    def assign(self, u=None):
        """
        Assign the model observation and compute the functional

        Arguments
        ---------
        u : :py:class:`dolfin.Function`
            The input to the model observation, e.g the displacement

        Returns
        -------
        functional : :py:class:`dolfin_adjoint.Function`
            A scalar representing the mismatch between model and data defined
            in the :meth:`OptimizationTarget.form` method.
        """
        # Assign model observation for dolfin-adjoint recording
        model = self.model(u)
        self.model_function.assign(model)

        # Assign data observation for dolfin-adjoint recording
        data = self.dolfin_observations[self.count]
        self.data_function.assign(data)

        form = self.form()

        dolfin_adjoint.solve(
            self._trial * self._test * dolfin.dx == self._test * form * dolfin.dx,
            self._functional,
        )

        if self.collect:
            self.collector['model'].append(dolfin.Vector(self.model_function.vector()))
            self.collector['data'].append(dolfin.Vector(self.data_function.vector()))
            self.collector['functional'].append(numpy_mpi.gather_broadcast(self._functional.vector().get_local())[0])

        return self.weight * self._functional

    @property
    def count(self):
        return max(self._count, 0)

    def __iter__(self):
        return self

    def __next__(self):
        self._count += 1
        if self.count == self.__len__:
            raise StopIteration
        else:
            return self
