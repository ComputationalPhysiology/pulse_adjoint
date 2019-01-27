from functools import partial
import logging
from collections import namedtuple
import dolfin
import dolfin_adjoint

import pulse
from pulse.iterate import iterate
from pulse import numpy_mpi, annotation
from pulse.dolfin_utils import list_sum, compute_meshvolume

from . import make_logger
from .reduced_functional import ReducedFunctional
from .optimal_control import OptimalControl
from .optimization_targets import OptimizationTarget
from .regularization import Regularization
from .model_observations import BoundaryObservation


logger = make_logger(__name__, 10)
forward_result = namedtuple("forward_result", "functional, functional_value, converged")


def tuplize(lst, instance, name):

    if isinstance(lst, instance):
        # There is only one target
        return (lst,)
    else:
        msg = (
            "{} must be a list or singleton of type " '"OptimizationTarget", got {}'
        ).format(name, lst)
        if not hasattr(lst, "__iter__"):
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

    Arguments
    ---------
    problem : :py:class:`pulse.MechanicsProblem`
        The mechanics problem containing the force-balance eqation,
        and geometry.
    targets : :py:class:`pulse_adjoint.OptimizationTarget` or list
        The optimzation targets either as a singleton or as a list.
    bcs : :py:class:`pulse_adjoint.BoundaryObservation` or list
        The observations that should be enforced on the boudary either
        as a singlton or as a list.
    control
    """

    def __init__(
        self,
        problem,
        targets,
        bcs,
        control,
        regularization=None,
        parameters=None
    ):

        self.problem = problem
        self.targets = tuplize(targets, OptimizationTarget, "targets")
        self.bcs = tuplize(bcs, BoundaryObservation, "bcs")

        self.control = control

        if regularization is None:
            self.regularization = Regularization.zero()
        else:
            msg = ('Expected regularization to be of type "pulse_adjoint.'
                   'Regularization" got {}'.format(type(regularization)))
            assert isinstance(regularization, Regularization), msg
            self.regularization = regularization

        self.parameters = Assimilator.default_parameters()
        if parameters is not None:
            self.parameters.update(**parameters)

        self.validate()

        # Keep the old state in case we are unable to iterate
        self.old_state = self.problem.state.copy(deepcopy=True,
                                                 name='Old state (pulse-adjoint)')
        self.old_control = self.control.copy(deepcopy=True,
                                             name='Old control (pulse-adjoint)')

    def validate(self):

        data_points = len(self.bcs[0].data)
        ref = self.bcs[0].bc
        for i in self.bcs:
            msg = (
                "Number of data points for BC {} is {} which does not "
                "match with {} which has {} data points"
                ""
            ).format(i.bc, len(i.data), ref, data_points)
            if len(i.data) != data_points:
                raise ValueError(msg)

        for i in self.targets:
            msg = (
                "Number of data points for target {} is {} which does not "
                "match with {} which has {} data point"
                ""
            ).format(i.model, len(i.observations), ref, data_points)
            if len(i.observations) != data_points:
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

    def iteration(self, control):
        """
        FIXME
        TODO: Make this more elegant
        """
        for count in range(self.data_points):

            # Stop the recording
            annotate = annotation.annotate

            bcs = [bc.bc.traction for bc in self.bcs]
            targets = [bc.data[count] for bc in self.bcs]

            # Iterate bc
            iterate(self.problem, bcs, targets)

            annotation.annotate = False

            # Iterate control
            prev_states, prev_controls = iterate(self.problem,
                                                 self.control,
                                                 control)


            self.problem.state.assign(prev_states[-1])
            self.control.assign(prev_controls[-1])
            # Continue recording
            annotation.annotate = annotate

            for b in self.bcs:
                b._count = count
                b.assign_bc()

            self.problem.solve()

            self.control.assign(control, annotate=annotate)
            self.problem.solve()

            u, p = dolfin.split(self.problem.state)

            for t in self.targets:
                t._count = count
                t.assign(u, annotate=annotate)

            yield count

    def reset_problem(self):

        annotate = annotation.annotate
        annotation.annotate = False
        self.control.assign(self.old_control)
        self.problem.reinit(self.old_state)

        for bc in self.bcs:
            bc.reset()

        for t in self.targets:
            t.reset()

        # Do one initial solve
        self.problem.solve()
        annotation.annotate = annotate

    def create_forward_problem(self):
        def forward(control, annotate=True):

            self.reset_problem()
            annotation.annotate = annotate


            # Start the clock
            dolfin_adjoint.adj_start_timestep(0.0)

            states = []
            functional_values = []

            functional = self.create_functional()
            functionals_time = []

            # functionals_time = [functional * dolfin_adjoint.dt[0.0]]

            gen = self.iteration(control)
            for count in gen:

                # Collect stuff
                states.append(dolfin.Vector(self.problem.state.vector()))
                functional_values.append(dolfin.assemble(functional))
                functionals_time.append(functional * dolfin_adjoint.dt[count + 1])
                dolfin_adjoint.adj_inc_timestep(count + 1, count == self.data_points)
                break

            return forward_result(
                functional=list_sum(functionals_time),
                functional_value=sum(functional_values),
                converged=True,
            )

        return forward

    def create_reduced_functional(self):

        forward_model = self.create_forward_problem()
        rd = ReducedFunctional(forward_model, self.control)
        rd(self.control)
        # 
        # dolfin_adjoint.replay_dolfin()
        # rd.derivative()
        # # exit()
        # dolfin_adjoint.adj_html("forward.html", "forward")
        # dolfin_adjoint.adj_html("adjoint.html", "adjoint")
        # exit()

        return rd

    def assimilate(self):
        """
        FIXME
        """
        loglevel = pulse.parameters["log_level"]
        # pulse.parameters["log_level"] = logging.WARNING

        # The functional we want to minimize
        rd = self.create_reduced_functional()
        
        # from IPython import embed; embed()
        # exit()
        # Create optimal control problem
        self.oc_problem = OptimalControl(min_value=0.1, max_value=10.0, tol=1e-3)
        x = numpy_mpi.gather_broadcast(self.control.vector().get_local())
        self.oc_problem.build_problem(rd, x)
        opt_results = self.oc_problem.solve()

        pulse.parameters["log_level"] = loglevel
