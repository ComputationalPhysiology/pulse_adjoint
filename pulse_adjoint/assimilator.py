from functools import partial
import logging
from collections import namedtuple
import dolfin
import dolfin_adjoint

import pulse
from pulse.iterate import iterate
from pulse import numpy_mpi
from pulse.dolfin_utils import list_sum, compute_meshvolume

from . import make_logger
from .dolfin_utils import ReducedFunctional
from .optimal_control import OptimalControl
from .optimization_targets import OptimizationTarget
from .model_observations import BoundaryObservation


logger = make_logger(__name__, 10)
forward_result = namedtuple('forward_result', 'functional, functional_value, converged')


def tuplize(lst, instance, name):

    if isinstance(lst, instance):
        # There is only one target
        return (lst,)
    else:
        msg = ('{} must be a list or singleton of type '
               '"OptimizationTarget", got {}').format(name, lst)
        if not hasattr(lst, '__iter__'):
            raise TypeError(msg)
        for l in lst:
            assert isinstance(l, instance), msg

        return lst

def make_optimization_results(opt_results, rd):
    pass


class Assimilator(object):
    """
    Class for assimilating clinical data with a mechanics problem
    by tuning some control parameters
    """

    def __init__(self, problem, targets, bcs, control, parameters=None):

        self.problem = problem
        self.targets = tuplize(targets, OptimizationTarget, 'targets')
        self.bcs = tuplize(bcs, BoundaryObservation, 'bcs')

        self.control = control

        self.parameters = Assimilator.default_parameters()
        if parameters is not None:
            self.parameters.update(**parameters)

        self.validate()

    def validate(self):

        data_points = len(self.bcs[0].data)
        ref = self.bcs[0].bc
        for i in self.bcs:
            msg = ('Number of data points for BC {} is {} which does not '
                   'match with {} which has {} data points'
                   '').format(i.bc, len(i.data), ref, data_points)
            if (len(i.data) != data_points):
                raise ValueError(msg)

        for i in self.targets:
            msg = ('Number of data points for target {} is {} which does not '
                   'match with {} which has {} data point'
                   '').format(i.model, len(i.observations), ref, data_points)
            if (len(i.observations) != data_points):
                raise ValueError(msg)
        self.data_points = data_points

        try:
            meshvol = self.problem.geometry.meshvol
        except AttributeError:
            meshvol = compute_meshvolume(self.problem.geometry.mesh)
        self._meshvol = dolfin_adjoint.Constant(meshvol)

    @staticmethod
    def default_parameters():
        """
        """
        opt_params = OptimalControl.default_parameters()

        return dict()

    def create_functional(self):

        functional_list = []
        for t in self.targets:
            functional_list = t.functional
        return list_sum(functional_list) / self._meshvol * dolfin.dx

    def iteration(self):
        """
        FIXME
        TODO: Make this more elegant
        """
        for count in range(self.data_points):

            controls = [bc.bc.traction for bc in self.bcs]
            targets = [bc.data[count] for bc in self.bcs]

            iterate(self.problem, controls, targets)

            u, p = dolfin.split(self.problem.state)

            for t in self.targets:
                t._count = count
                t.assign(u)

            yield count

    def reset_problem(self):

        self.problem.state.vector().zero()
        for bc in self.bcs:
            bc._count = -2
            bc.assign_bc()

        for t in self.targets:
            t._count = -1
            t.assign(None)

    def create_forward_problem(self):

        def forward(control, annotate=True):

            self.reset_problem()

            # Start the clock
            dolfin_adjoint.adj_start_timestep(0.0)

            states = []
            functional_values = []

            functional = self.create_functional()

            functionals_time = [functional * dolfin_adjoint.dt[0.0]]

            gen = self.iteration()
            for count in gen:

                # Collect stuff
                states.append(dolfin.Vector(self.problem.state.vector()))
                functional_values.append(dolfin.assemble(functional))
                functionals_time.append(functional * dolfin_adjoint.dt[count])
                dolfin_adjoint.adj_inc_timestep(count,
                                                count == (self.data_points - 1))

            return forward_result(functional=list_sum(functionals_time),
                                  functional_value=sum(functional_values),
                                  converged=True)

        return forward

    def create_reduced_functional(self):

        forward_model = self.create_forward_problem()
        rd = ReducedFunctional(forward_model, self.control)
        # rd(self.control)

        return rd

    def assimilate(self):
        """
        FIXME
        """
        loglevel = pulse.parameters['log_level']
        pulse.parameters['log_level'] = logging.WARNING


        # The functional we want to minimize
        rd = self.create_reduced_functional()

        # from IPython import embed; embed()
        # exit()
        # Create optimal control problem
        self.oc_problem = OptimalControl(min_value=0.1, max_value=10.0, tol=1e-3)
        x = numpy_mpi.gather_broadcast(self.control.vector().get_local())
        self.oc_problem.build_problem(rd, x)
        opt_results = self.oc_problem.solve()

        pulse.parameters['log_level'] = loglevel
