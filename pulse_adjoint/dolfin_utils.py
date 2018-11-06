import math
import dolfin
import dolfin_adjoint

from pulse.utils import make_logger

logger = make_logger(__name__, 10)


class ReducedFunctional(dolfin_adjoint.ReducedFunctional):
    """
    A modified reduced functional of the `dolfin_adjoint.ReducedFuctionl`

    *Parameters*

    for_run: callable
        The forward model, which can be called with the control parameter
        as first argument, and a boolean as second, indicating that
        annotation is on/off.
    paramvec: :py:class`dolfin_adjoint.function`
        The control parameter
    scale: float
        Scale factor for the functional
    relax: float
        Scale factor for the derivative. Note the total scale factor for the
        derivative will be scale*relax


    """
    def __init__(self, for_run, control, scale=1.0,
                 relax=1.0, verbose=False, log_level=dolfin.INFO):

        self.for_run = for_run
        self.control = control
        self.scale = scale
        self.derivative_scale = relax
        self.log_level = log_level
        self.verbose = verbose

        self.reset()

    def __call__(self, value, return_fail=False):

        # logger.debug("\nEvaluate functional...")
        dolfin_adjoint.adj_reset()

        self.count += 1

        paramvec_new = dolfin_adjoint.Function(self.paramvec.function_space(),
                                               name="new control")

        if isinstance(value, (dolfin.Function, dolfin_adjoint.Function,
                              RegionalParameter, MixedParameter)):
            paramvec_new.assign(value)
            
        elif isinstance(value, float) or isinstance(value, int):
            numpy_mpi.assign_to_vector(paramvec_new.vector(),
                                       np.array([value]))
            
        elif isinstance(value, dolfin_adjoint.enlisting.Enlisted):
            val_delisted = dolfin_adjoint.delist(value, self.controls)
            paramvec_new.assign(val_delisted)

        else:
            numpy_mpi.assign_to_vector(paramvec_new.vector(),
                                       numpy_mpi.gather_broadcast(value))

        logger.debug(utils.Text.yellow("Start annotating"))
        dolfin.parameters["adjoint"]["stop_annotating"] = False

        if self.verbose:

            arr = numpy_mpi.gather_broadcast(paramvec_new.vector().array())
            msg = ("\nCurrent value of control:"
                   "\n\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}"
                   "\n\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\t{:>8d}\t{:>8d}") \
                .format("Min",
                        "Mean",
                        "Max",
                        "argmin",
                        "argmax",
                        np.min(arr),
                        np.mean(arr),
                        np.max(arr),
                        np.argmin(arr),
                        np.argmax(arr))

            logger.info(msg)
        # Change loglevel to avoid too much printing
        change_log_level = (self.log_level == dolfin.INFO) \
            and not self.verbose

        if change_log_level:
            logger.setLevel(dolfin.WARNING)

        t = dolfin.Timer("Forward run")
        t.start()

        logger.debug("\nEvaluate forward model")

        self.for_res, crash = self.for_run(paramvec_new, True)

        for_time = t.stop()
        logger.debug(("Evaluating forward model done. "
                      "Time to evaluate = {} seconds".format(for_time)))
        self.forward_times.append(for_time)

        if change_log_level:
            logger.setLevel(self.log_level)

        if self.first_call:
            # Store initial results
            self.ini_for_res = self.for_res
            self.first_call = False

            # Some printing
            logger.info(utils.print_head(self.for_res))

        control = dolfin_adjoint.Control(self.paramvec)

        dolfin_adjoint.ReducedFunctional.__init__(
            self, dolfin_adjoint.Functional(self.for_res["total_functional"]),
            control)

        if crash:
            # This exection is thrown if the solver uses more than x steps.
            # The solver is stuck, return a large value so it does not get
            # stuck again
            logger.warning(
                utils.Text.red(("Iteration limit exceeded."
                                " Return a large value of the functional")))
            # Return a big value, and make sure to increment the big value
            # so the the next big value is different from the current one.
            func_value = np.inf
            self.nr_crashes += 1

        else:
            func_value = self.for_res["func_value"]

        # grad_norm = None if len(self.grad_norm_scaled) == 0 \
                    # else self.grad_norm_scaled[-1]

        self.func_values_lst.append(func_value * self.scale)
        self.controls_lst.append(dolfin.Vector(paramvec_new.vector()))

        logger.debug(utils.Text.yellow("Stop annotating"))
        dolfin.parameters["adjoint"]["stop_annotating"] = True

        self.print_line()

        if return_fail:
            return self.scale*func_value, crash

        return self.scale * func_value

    def reset(self):

        logger.setLevel(self.log_level)
        if not hasattr(self, "ini_for_res"):

            self.cache = None
            self.first_call = True
            self.nr_crashes = 0
            self.count = 0
            self.nr_der_calls = 0
            self.func_values_lst = []
            self.controls_lst = []
            self.forward_times = []
            self.backward_times = []
            self.grad_norm = []
            self.grad_norm_scaled = []
        else:
            if len(self.func_values_lst):
                self.func_values_lst.pop()
            if len(self.controls_lst):
                self.controls_lst.pop()
            if len(self.grad_norm):
                self.grad_norm.pop()
            if len(self.grad_norm_scaled):
                self.grad_norm_scaled.pop()

    def print_line(self):
        grad_norm = None if len(self.grad_norm_scaled) == 0 \
                    else self.grad_norm_scaled[-1]

        func_value = self.for_res["func_value"]

        logger.info(utils.print_line(self.for_res, self.iter,
                                     grad_norm, func_value))

    def derivative(self, *args, **kwargs):

        logger.debug("\nEvaluate gradient...")
        self.nr_der_calls += 1
        
        t = dolfin.Timer("Backward run")
        t.start()

        out = dolfin_adjoint.ReducedFunctional.derivative(self, forget=False)
        back_time = t.stop()
        logger.debug(("Evaluating gradient done. "
                      "Time to evaluate = {} seconds".format(back_time)))
        self.backward_times.append(back_time)

        for num in out[0].vector().array():
            if math.isnan(num):
                raise Exception("NaN in adjoint gradient calculation.")

        # Multiply with some small number to that we take smaller steps
        gathered_out = numpy_mpi.gather_broadcast(out[0].vector().array())

        self.grad_norm.append(np.linalg.norm(gathered_out))
        self.grad_norm_scaled.append(np.linalg.norm(gathered_out)
                                     * self.scale * self.derivative_scale)
        logger.debug(("|dJ|(actual) = {}\t"
                      "|dJ|(scaled) = {}").format(self.grad_norm[-1],
                                                  self.grad_norm_scaled[-1]))
        return self.scale*gathered_out*self.derivative_scale


