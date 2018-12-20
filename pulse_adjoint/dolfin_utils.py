import math
import dolfin
import dolfin_adjoint
import numpy as np


from pulse import MixedParameter, RegionalParameter, numpy_mpi
from . import make_logger, Text, annotation

logger = make_logger(__name__, 10)


class ReducedFunctional(dolfin_adjoint.ReducedFunctional):
    """
    A modified reduced functional of the `dolfin_adjoint.ReducedFuctionl`

    *Parameters*

    for_run: callable
        The forward model, which can be called with the control parameter
        as first argument, and a boolean as second, indicating that
        annotation is on/off.
    control: :py:class`dolfin_adjoint.function`
        The control parameter
    scale: float
        Scale factor for the functional
    relax: float
        Scale factor for the derivative. Note the total scale factor for the
        derivative will be scale*relax


    """

    def __init__(
        self,
        forward_model,
        control,
        scale=1.0,
        derivate_scale=1.0,
        verbose=False,
        log_level=dolfin.INFO,
    ):

        self.forward_model = forward_model
        self.control = control
        self.scale = scale
        self.derivative_scale = derivate_scale
        self.log_level = log_level
        self.verbose = verbose
        self.reset()

    def __call__(self, value):

        logger.debug("\nEvaluate functional...")

        # Start recording
        dolfin_adjoint.adj_reset()
        annotation.annotate = False

        self.collector['count'] += 1
        self.assign_control(value)

        if self.verbose:

            arr = numpy_mpi.gather_broadcast(self.control.vector().array())
            msg = (
                "\nCurrent value of control:"
                "\n\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}"
                "\n\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\t{:>8d}\t{:>8d}"
            ).format(
                "Min",
                "Mean",
                "Max",
                "argmin",
                "argmax",
                np.min(arr),
                np.mean(arr),
                np.max(arr),
                np.argmin(arr),
                np.argmax(arr),
            )

            logger.info(msg)

        # Change loglevel to avoid too much printing
        change_log_level = (self.log_level == dolfin.INFO) and not self.verbose

        if change_log_level:
            logger.setLevel(dolfin.WARNING)

        t = dolfin.Timer("Forward run")
        t.start()

        logger.debug("\nEvaluate forward model")

        self.forward_result = self.forward_model(self.control, annotate=True)
        crash = not self.forward_result.converged

        forward_time = t.stop()
        self.collector['forward_times'].append(forward_time)
        logger.debug(
            (
                "Evaluating forward model done. "
                "Time to evaluate = {} seconds".format(forward_time)
            )
        )

        if change_log_level:
            logger.setLevel(self.log_level)

        if self.first_call:
            # Store initial results
            self.collector['initial_results'] = self.forward_result
            self.first_call = False

            # Some printing
            # logger.info(utils.print_head(self.for_res))

        control = dolfin_adjoint.Control(self.control)

        dolfin_adjoint.ReducedFunctional.__init__(
            self, dolfin_adjoint.Functional(self.forward_result.functional),
            control
        )

        if crash:
            # This exection is thrown if the solver uses more than x steps.
            # The solver is stuck, return a large value so it does not get
            # stuck again
            logger.warning(
                Text.red(
                    (
                        "Iteration limit exceeded. "
                        "Return a large value of the functional"
                    )
                )
            )
            # Return a big value, and make sure to increment the big value
            # so the the next big value is different from the current one.
            func_value = np.inf
            self.collector['nr_crashes'] += 1

        else:
            func_value = self.forward_result.functional_value

        # grad_norm = None if len(self.grad_norm_scaled) == 0 \
        # else self.grad_norm_scaled[-1]

        self.collector['functional_values'].append(func_value * self.scale)
        self.collector['controls'].append(dolfin.Vector(self.control.vector()))

        logger.debug(Text.yellow("Stop annotating"))
        annotation.annotate = False

        self.print_line()
        return self.scale * func_value

    def assign_control(self, value):
        """
        Assign value to control parameter
        """
        control_new = dolfin_adjoint.Function(
            self.control.function_space(), name="new control"
        )

        if isinstance(
            value,
            (
                dolfin.Function,
                dolfin_adjoint.Function,
                RegionalParameter,
                MixedParameter,
            ),
        ):
            control_new.assign(value)

        elif isinstance(value, float) or isinstance(value, int):
            numpy_mpi.assign_to_vector(control_new.vector(), np.array([value]))

        elif isinstance(value, dolfin_adjoint.enlisting.Enlisted):
            val_delisted = dolfin_adjoint.delist(value, self.controls)
            control_new.assign(val_delisted)

        else:
            numpy_mpi.assign_to_vector(
                control_new.vector(), numpy_mpi.gather_broadcast(value)
            )

        self.control.assign(control_new)

    def reset(self):

        logger.setLevel(self.log_level)
        if not hasattr(self, "ini_for_res"):
            self.first_call = True
            self.collector = dict(nr_crashes=0,
                                  count=0,
                                  nr_derivative_calls=0,
                                  functional_values=[],
                                  controls=[],
                                  forward_times=[],
                                  backward_times=[],
                                  gradient_norm=[],
                                  gradient_norm_scaled=[])
        else:

            for key in ['functional_values', 'controls',
                        'gradient_norm', 'gradient_norm_scaled']:
                v = self.collector.get(key, [])
                if len(v) > 0:
                    v.pop()

    def print_line(self):
        grad_norm = (
            None if len(self.collector['gradient_norm_scaled']) == 0
            else self.collector['gradient_norm_scaled'][-1]
        )

        func_value = self.forward_result.functional_value
        print('Gradient = {}, Func value = {}'.format(grad_norm, func_value))
        # logger.info(utils.print_line(self.for_res, self.iter, grad_norm, func_value))

    def derivative(self, *args, **kwargs):

        logger.debug("\nEvaluate gradient...")
        self.collector['nr_derivative_calls'] += 1

        t = dolfin.Timer("Backward run")
        t.start()

        out = dolfin_adjoint.ReducedFunctional.derivative(self, forget=False)
        back_time = t.stop()
        logger.debug(
            (
                "Evaluating gradient done. "
                "Time to evaluate = {} seconds".format(back_time)
            )
        )
        self.collector['backward_times'].append(back_time)

        for num in out[0].vector().array():
            if math.isnan(num):
                raise Exception("NaN in adjoint gradient calculation.")

        # Multiply with some small number to that we take smaller steps
        gathered_out = numpy_mpi.gather_broadcast(out[0].vector().array())

        self.collector['gradient_norm'].append(np.linalg.norm(gathered_out))
        self.collector['gradient_norm_scaled'].append(
            np.linalg.norm(gathered_out) * self.scale * self.derivative_scale
        )
        logger.debug(
            ("|dJ|(actual) = {}\t" "|dJ|(scaled) = {}").format(
                self.collector['gradient_norm'][-1],
                self.collector['gradient_norm_scaled'][-1]
            )
        )
        return self.scale * gathered_out * self.derivative_scale
