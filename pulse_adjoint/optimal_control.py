#!/usr/bin/env python
r"""
This script implements that various opimtimization options
that can be used to solve the optimal control problem

.. math::

   \min_{m} \mathcal{J}(\mathbf{u},p,m)

   \mathrm{subject}\;\mathrm{to:} \: \delta\Pi(\mathbf{u},p,m) = 0


**Example of usage**::

  # Suppose you allready have initialized you reduced functional (`J`)
  # with the control parameter (`paramvec`)
  # Look at run_optimization.py to see how to do this.

  # Initialize the paramters
  params = setup_application_parameters()

  # params["Optimization_parameters"]["opt_type"] = "pyOpt_slsqp"
  # params["Optimization_parameters"]["opt_type"] = "scipy_l-bfgs-b"
  params["Optimization_parameters"]["opt_type"] = "scipy_slsqp"

  # Create the optimal control problem
  oc_problem = OptimalControl()
  # Build the problem
  oc_problem.build_problem(params, J, paramvec)
  # Solve the optimal control problem
  J, opt_result = oc_problem.solve()

"""
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS:
# post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
from collections import namedtuple

import numpy as np
from dolfin import Timer

from . import make_logger

logger = make_logger(__name__, 10)

optimization_results = namedtuple(
    "optimization_results", "initial_control, optimal_control, run_time"
)

try:
    import scipy
    from scipy.optimize import minimize as scipy_minimize
    from scipy.optimize import minimize_scalar as scipy_minimize_1d

    has_scipy = True
    from distutils.version import StrictVersion

    has_scipy016 = StrictVersion(scipy.version.version) >= StrictVersion("0.16")

except ImportError:
    has_scipy = False
    has_scipy016 = False

try:
    import pyipopt

    has_pyipopt = True

except ImportError:
    has_pyipopt = False

try:
    import moola  # noqa: F401

    has_moola = True
except ImportError:
    has_moola = False

try:
    import pyOpt

    has_pyOpt = True

except ImportError:
    has_pyOpt = False

opt_import = [has_scipy, has_moola, has_pyOpt, has_pyipopt]


class MyCallBack(object):
    """pass a custom callback function

    This makes sure it's being used.
    """

    def __init__(self, J, tol, max_iter):

        self.ncalls = 0
        self.J = J
        self.opt_funcvalues = []

    def __call__(self, x):

        self.ncalls += 1

        # grad_norm = (
        #     None if len(self.J.grad_norm_scaled) == 0 else self.J.grad_norm_scaled[-1]
        # )
        #
        # func_value = self.J.forwaJ_result.functional_value
        # self.opt_funcvalues.append(func_value)
        # self.J.opt_funcvalues = self.opt_funcvalues

        # logger.info(print_line(self.J.for_res, self.ncalls, grad_norm, func_value))


def minimize_1d(f, x0, **kwargs):
    """Minimize functional with one variable using the
    brent algorithm from scpiy.

    Parameters
    ----------
    f: callable
        Objective functional
    x0: float
        initial guess

    *Returns*

    res: dict
        Scipy results from the opimization

    """

    return scipy_minimize_1d(f, **kwargs)


def get_ipopt_options(rd, lb, ub, tol, max_iter, **kwargs):
    """Get options for IPOPT module (interior point algorithm)

    See `<https://projects.coin-or.org/Ipopt>`

    Parameters
    ----------
    rd : :py:class`dolfin_adjoint.ReducedFunctional`
            The reduced functional
    lb : list
        Lower bound on the control
    ub : list
        Upper bound on the control
    tol : float
        Tolerance
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    nlp : ipopt instance
        A nonlinear ipopt problem
    """
    ncontrols = len(ub)
    # nconstraints = 0
    empty = np.array([], dtype=float)
    clb = empty
    cub = empty
    # The constraint function, should do nothing

    def fun_g(x, user_data=None):
        return empty

    # The constraint Jacobian
    def jac_g(x, flag, user_data=None):
        if flag:
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)
            return (rows, cols)
        else:
            return empty

    J = rd.__call__
    dJ = rd.derivative

    nlp = pyipopt.create(
        ncontrols,  # length of control vector
        lb,  # lower bounds on control vector
        ub,  # upper bounds on control vector
        0,  # number of constraints
        clb,  # lower bounds on constraints,
        cub,  # upper bounds on constraints,
        0,  # number of nonzeros in the constraint Jacobian
        0,  # number of nonzeros in the Hessian
        J,  # to evaluate the functional
        dJ,  # to evaluate the gradient
        fun_g,  # to evaluate the constraints
        jac_g,
    )  # to evaluate the constraint Jacobian

    pyipopt.set_loglevel(1)
    return nlp


def get_moola_options(*args, **kwargs):
    """Get options for moola module.

    See `<https://github.com/funsim/moola>`

    .. note::

       This is not working

    """

    # problem = MoolaOptimizationProblem(J)

    # paramvec_moola = moola.DolfinPrimalVector(paramvec)
    # solver = moola.NewtonCG(problem, paramvec_moola, options={'gtol': 1e-9,
    #                                                           'maxiter': 20,
    #                                                           'display': 3,
    #                                                           'ncg_hesstol': 0})

    # solver = moola.BFGS(problem, paramvec_moola, options={'jtol': 0,
    #                                                       'gtol': 1e-9,
    #                                                       'Hinit': "default",
    #                                                       'maxiter': 100,
    #                                                     'mem_lim': 10})

    # solver = moola.NonLinearCG(problem, paramvec_moola, options={'jtol': 0,
    #                                                          'gtol': 1e-9,
    #                                                          'Hinit': "default",
    #                                                          'maxiter': 100,
    #                                                          'mem_lim': 10})

    raise NotImplementedError


def get_scipy_options(J, method, lb, ub, tol, max_iter, **kwargs):
    """Get options for scipy module

    See `<https://docs.scipy.org/doc/scipy-0.18.1/reference/optimize.html>`

    Parameters
    ----------
    J : :py:class`dolfin_adjoint.ReducedFunctional`
            The reduced functional
    method : str
        Which optimization algorithm 'LBFGS' or 'SLSQP'.
    lb : list
        Lower bound on the control
    ub : list
        Upper bound on the control
    tol : float
        Tolerance
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    options : dict
        The options to be passed to the scipy optimization

    """

    def lowerbound_constraint(m):
        return m - lb

    def upperbound_constraint(m):
        return ub - m

    cons = (
        {"type": "ineq", "fun": lowerbound_constraint},
        {"type": "ineq", "fun": upperbound_constraint},
    )

    if not has_scipy016 and method == "slsqp":
        callback = None
    else:
        callback = MyCallBack(J, tol, max_iter)

    options = {
        "method": method,
        "jac": J.derivative,
        "tol": tol,
        "callback": callback,
        "options": {
            "disp": kwargs.pop("disp", False),
            "iprint": kwargs.pop("iprint", 2),
            "ftol": tol,
            "maxiter": max_iter,
        },
    }

    if method == "slsqp":
        options["constraints"] = cons
    else:
        options["bounds"] = zip(lb, ub)

    return options


def get_pyOpt_options(J, method, lb, ub, tol, max_iter, **kwargs):
    """Get options for pyOpt module

    See `<http://www.pyopt.org>`

    Parameters
    ----------
    method : str
        Which optimization algorithm `not working` SLSQP will be chosen.
    J : :py:class`dolfin_adjoint.ReducedFunctional`
            The reduced functional
    lb : list
        Lower bound on the control
    ub : list
        Upper bound on the control
    tol : float
        Tolerance
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    opt : tuple
        The optimization solver and the options, (solver, options)

    """

    def obj(x):
        f, fail = J(x, True)
        g = []

        return f, g, fail

    def grad(x, f, g):
        fail = False
        try:
            dj = J.derivative()
        except Exception as ex:
            logger.warning(ex)
            fail = True

        # Contraints gradient
        gJac = np.zeros(len(x))

        # logger.info("j = %f\t\t|dJ| = %f" % (f[0], np.linalg.norm(dj)))
        return np.array([dj]), gJac, fail

    # Create problem
    opt_prob = pyOpt.Optimization("Problem", obj)

    # Assign objective
    opt_prob.addObj("J")

    # Assign design variables (bounds)
    opt_prob.addVarGroup(
        "variables", kwargs["nvar"], type="c", value=kwargs["m"], lower=lb, upper=ub
    )

    opt = pyOpt.pySLSQP.SLSQP()
    opt.setOption("ACC", tol)
    opt.setOption("MAXIT", max_iter)
    opt.setOption("IPRINT", -1)
    opt.setOption("IFILE", "")

    options = {"opt_problem": opt_prob, "sens_type": grad}

    return opt, options


class OptimalControl(object):
    """
    A class used for solving an optimal control problem
    For possible parameters in the constructor see
    :py:meth:`OptimalControl.default_parameters´
    """

    def __init__(self, **parameters):

        self.parameters = OptimalControl.default_parameters()

        self.parameters.update(**{k: v for k, v in parameters.items() if v is not None})

    @staticmethod
    def default_parameters():
        return dict(
            max_value=1.0,
            min_value=0.0,
            max_iter=100,
            tol=1e-6,
            opt_lib="scipy",
            method="slsqp",
            method_1d="bounded",
        )

    def build_problem(self, J, x):
        """Build optimal control problem

        Parameters
        ----------
        J : callable
            The objective function that you want to minimize.
            It should be able to be called on the control, i.e
            you should be able to do :math:`J(x)` where :math:`x`
            is the given control parameter.
            If you use a gradient-based optimzation method it must
            also implement an instance method called `derivative`, i.e
            it should be possible to do :math:`J.derivative(x)`.
        control : np.ndarray
            Control parameter

        """
        msg = "No supported optimization module installed"
        assert any(opt_import), msg

        # x = gather_broadcast(control.vector().get_local())
        nvar = len(x)
        self.x = x
        self.initial_control = np.copy(x)
        self.J = J

        self.parameters["nvar"] = nvar
        self.parameters["lb"] = np.array([self.parameters["min_value"]] * nvar)
        self.parameters["ub"] = np.array([self.parameters["max_value"]] * nvar)

        self.parameters["bounds"] = list(
            zip(self.parameters["lb"], self.parameters["ub"])
        )

        self._set_options()

        logger.info("Building optimal control problem")
        # msg = (
        #     "\n\tNumber of variables:\t{}".format(nvar)
        #     + "\n\tLower bound:\t{}".format(np.min(lb))
        #     + "\n\tUpper bound:\t{}".format(np.max(ub))
        #     + "\n\tTolerance:\t{}".format(tol)
        #     + "\n\tMaximum iterations:\t{}".format(max_iter)
        #     + "\n\tOptimization algoritmh:\t{}\n".format(self.opt_type)
        # )
        # logger.info(msg)

    def _set_options(self):

        module = self.parameters["opt_lib"]

        if module == "scipy":
            assert has_scipy, "Scipy not installed"
            if self.parameters["nvar"] == 1:
                self.options = {
                    "method": self.parameters["method_1d"],
                    "bounds": self.parameters["bounds"][0],
                    "tol": self.parameters["tol"],
                    "options": {"maxiter": self.parameters["max_iter"]},
                }
            else:
                self.options = get_scipy_options(self.J, **self.parameters)

        elif module == "moola":
            assert has_moola, "Moola not installed"
            self.solver = get_moola_options(self.J, **self.parameters)

        elif module == "pyOpt":
            assert has_pyOpt, "pyOpt not installed"
            self.problem, self.options = get_pyOpt_options(self.J, **self.parameters)

        elif module == "ipopt":
            assert has_pyipopt, "IPOPT not installed"
            self.solver = get_ipopt_options(self.J, **self.parameters)

        else:
            msg = (
                "Unknown optimizatin type {}. "
                "Define the optimization type as 'module-method', "
                "where module is e.g scipy, pyOpt and methos is "
                "eg slsqp."
            )
            raise ValueError(msg)

    def solve(self):
        """
        Solve optmal control problem

        """

        # msg = "You need to build the problem before solving it"
        # assert hasattr(self, "opt_type"), msg

        module = self.parameters["opt_lib"]

        logger.info("Starting optimization")
        # logger.info(
        #     "Scale: {}, \nDerivative Scale: {}".format(
        #         self.J.scale, self.J.derivative_scale
        #     )
        # )
        # logger.info(
        #     "Tolerace: {}, \nMaximum iterations: {}\n".format(self.tol, self.max_iter)
        # )

        t = Timer()
        t.start()

        if self.parameters["nvar"] == 1:

            res = minimize_1d(self.J, self.x[0], **self.options)
            x = res["x"]

        else:

            if module == "scipy":
                res = scipy_minimize(self.J, self.x, **self.options)
                x = res["x"]

            elif module == "pyOpt":
                obj, x, d = self.problem(**self.options)

            elif module == "moola":
                sol = self.solver.solve()
                x = sol["control"].data

            elif module == "ipopt":
                x = self.solver.solve(self.x)

            else:
                msg = (
                    "Unknown optimizatin type {}. "
                    "Define the optimization type as 'module-method', "
                    "where module is e.g scipy, pyOpt and methos is "
                    "eg slsqp."
                )
                raise ValueError(msg)

        run_time = t.stop()
        # opt_result = {}
        # opt_result['initial_contorl'] = self.initial_control
        # opt_result["optimal_control"] = x
        # opt_result["run_time"] = run_time
        # opt_result["nfev"] = self.J.iter
        # opt_result["nit"] = self.J.iter
        # opt_result["njev"] = self.J.nr_der_calls
        # opt_result["ncrash"] = self.J.nr_crashes
        # opt_result["controls"] = self.J.controls_lst
        # opt_result["func_vals"] = self.J.func_values_lst
        # opt_result["forwaJ_times"] = self.J.forwaJ_times
        # opt_result["backwaJ_times"] = self.J.backwaJ_times
        # opt_result["grad_norm"] = self.J.grad_norm
        return optimization_results(
            initial_control=self.initial_control, optimal_control=x, run_time=run_time
        )
