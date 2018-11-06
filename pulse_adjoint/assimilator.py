import dolfin
import dolfin_adjoint
# from optimization_targets_new import RegionalStrainTarget, VolumeTarget


# from . import config
from .dolfin_utils import ReducedFunctional
from .optimization_targets import (RegionalStrainTarget,
                                   Regularization, VolumeTarget)
from pulse.dolfin_utils import list_sum, get_cavity_volume_form

from pulse.utils import make_logger, get_lv_marker

from pulse import annotate
from pulse.iterate import iterate
logger = make_logger(__name__, 10)


def get_optimization_targets(problem, control, params):
    """FIXME! briefly describe function
    """
    logger.debug("Get optimization targets")

    p = params

    targets = {}
    if p["regularization"] > 0.0:

        targets["regularization"] \
            = Regularization(problem.geometry.mesh,
                             spacestr,
                             p["regularization"],
                             mshfun=problem.geometry.cfun)

    if p["volume"] > 0.0:
        logger.debug("Load volume target")
        marker = get_lv_marker(problem.geometry)
        logger.debug(("Make surface meausure for LV endo "
                      "with marker {}").format(marker))

        ds = problem.geometry.ds(marker)

        logger.debug("Load VolumeTarget")
        targets["volume"] = VolumeTarget(mesh=problem.geometry.mesh,
                                         dmu=ds, chamber="LV",
                                         approx=params["volume_approx"])

    if p["rv_volume"] > 0.0:
        logger.debug("Load RV volume target")
        marker = problem.geometry.markers.get("ENDO_RV", [None])[0]
        if marker is None:
            logger.warning(('No marker for RV endo. Can not assimilate '
                            'RV volume. Consider setting '
                            "parameters['optimization_weights']"
                            "['rv_volume'] = 0"))

        else:
            logger.debug(("Make surface meausure for RV endo "
                          "with marker {}").format(marker))

            ds = problem.geometry.ds(marker)

            logger.debug("Load VolumeTarget")
            targets["rv_volume"] = VolumeTarget(mesh=problem.geometry.mesh,
                                                dmu=ds, chamber="RV",
                                                approx=params["volume_approx"])

    # if p["regional_strain"] > 0.0:

    #     logger.debug("Load regional strain target")
    #     dx = problem.geometry.dx

    #     load_displacemet = (params["unload"] and
    #                         not params["strain_reference"] == "unloaded") or \
    #                        (not params["unload"]
    #                         and params["strain_reference"] == "ED")

    #     if load_displacemet and params["phase"] == config.PHASES[1]:
    #         # We need to recompute strains wrt reference as diastasis

    #         logger.debug(("Load displacment for recomputing strain with "
    #                       "respect to different reference"))
            
    #         if params["strain_reference"] == "0":
    #             group = "1"
    #         else:
    #             pass
    #             # strain reference = "ED"

    #             #FIXME!! passive filling duration
    #             # if params["unload"]:
    #             #     group = str(solver_parameters["passive_filling_duration"])
    #             # else:
    #             #     group = str(solver_parameters["passive_filling_duration"]-1)

    #         u = dolfin_adjoint.Function(problem.state_space.sub(0).collapse())

    #         logger.debug("Load displacement from state number {}.".format(group))
    #         with dolfin.HDF5File(dolfin.mpi_comm_world(),
    #                              params["sim_file"], 'r') as h5file:

    #             # Get previous state
    #             group = "/".join([params["h5group"],
    #                               config.PASSIVE_INFLATION_GROUP,
    #                               "displacement", group])
    #             h5file.read(u, group)

    #         if params["strain_approx"] in ["project", "interpolate"]:

    #             V = dolfin.VectorFunctionSpace(problem.geometry.mesh,
    #                                            "CG", 1)
    
    #             if params["strain_approx"] == "project":
    #                 logger.debug("Project displacement")
    #                 u = dolfin_adjoint.project(u, V, name="u_project")
    #                 logger.debug("Interpolate displacement")
    #                 u = dolfin_adjoint.interpolate(u, V, name="u_interpolate")

    #         F_ref = kinematics.DeformationGradient(u)

    #     else:
    #         logger.debug("Do not recompute strains with respect than difference reference")
    #         F_ref = dolfin.Identity(3)

    #     logger.debug("Get RegionalStrainTarget")
    #     targets["regional_strain"] = \
    #         RegionalStrainTarget(mesh=problem.geometry.mesh,
    #                              crl_basis=problem.geometry.crl_basis,
    #                              dmu=dx,
    #                              weights=None,
    #                              tensor=params["strain_tensor"],
    #                              F_ref=F_ref,
    #                              approx=params["strain_approx"],
    #                              map_strain=params["map_strain"])

    return targets


def load_targets(problem, data, control, params):

    annotate.annotate = False

    targets = get_optimization_targets(problem, control, params)

    if problem.geometry.is_biv:
        pressure = zip(data.pressure, data.RVP)
    else:
        pressure = data.pressure

    return targets, pressure


def make_functional(targets, weights=None):

    if weights is None:
        weights = {}

    # Get the functional value of each term in the functional
    func_lst = []
    for k, v in targets.items():
        w = weights.get(k, 1.0)
        
        func_lst.append(dolfin_adjoint.Constant(w) * v.get_functional())

    # Collect the terms in the functional
    functional = list_sum(func_lst)

    return functional


def update_targets(self, it, u, m, annotate=False):

    for key, val in self.target_params.iteritems():

        if val:
            self.optimization_targets[key].next_target(it,
                                                       annotate=annotate)
            self.optimization_targets[key].assign_simulated(u)
            self.optimization_targets[key].assign_functional()
            self.optimization_targets[key].save()

    self.regularization.assign(m, annotate=annotate)
    self.regularization.save()

    
def forward(problem, data, control, params):

    optimization_targets, bcs \
        = load_targets(problem, data, control, params)

    # dolfin_adjoint.continue_annotation()

    control = dolfin_adjoint.Control(problem.material.activation)
    # Start the clock
    # dolfin_adjoint.adj_start_timestep(0.0)

    states = []
    functional_values = []
    functionals_time = []

    functional = make_functional(optimization_targets)

    for it, bc in enumerate(bcs):

        # iterate(problem, problem.bcs.neumann[0].traction,
        #         dolfin_adjoint.Constant(bc))
        problem.bcs.neumann[0].traction.assign(dolfin_adjoint.Constant(bc))
        problem.solve()
        u, p = dolfin.split(problem.state)

        for k, v in optimization_targets.items():

            d = data[k, it]

            vol_form = get_cavity_volume_form(problem.geometry.mesh,
                                              u=u)
            # vol = problem.geometry.cavity_volume(u=u)
            ds = problem.geometry.ds(problem.geometry.markers["ENDO"][0])
            J = dolfin_adjoint.assemble(((vol_form - d)**2) * ds)
            grad = dolfin_adjoint.compute_gradient(J, control)
            from IPython import embed; embed()
            exit()
            # v.assign_target(d, annotate=True)
            # v.assign_simulated(u)
            # v.assign_functional()
            # v.assign_functional()

        # functional_times.append( 
        functional_values.append(functional)

    total_functional = dolfin_adjoint.assemble(list_sum(functional_values) * dolfin.dx)
    print("Compute gradient")
    grad = dolfin_adjoint.compute_gradient(total_functional, control)
    from IPython import embed; embed()
    exit()
    


class Assimilator(object):
    """
    Class for assimilating clinical data with a mechanics problem
    by tuning some control parameters
    """
    def __init__(self, problem, data, control, parameters=None):

        self.problem = problem
        self.data = data
        self.control = control

        self.parameters = Assimilator.default_parameters()
        if parameters is not None:
            self.parameters.update(**parameters)

    @staticmethod
    def default_parameters():
        """
        strain_tensor in ['gradu', 'E', 'almansi']
        u_approximation_x in ['original', 'project', 'interpolate']
        
        """
        return dict()# (strain_tensor='E',
                   #  u_approximation_strain='original',
                   #  u_approximation_volume='original')


    def create_forward_problem(self):

        params = dict(volume=1, rv_volume=0.0,
                      regularization=0.0,
                      volume_approx='original')
        return forward(self.problem, self.data, self.control, params)
        

    def create_reduced_functional(self):

        forward = self.create_forward_problem()
        rd = ReducedFunctional(forward, self.control)

        return rd
    
    def assimilate(self):
        """
        FIXME
        """
        rd = self.create_reduced_functional()
        
        
        
        
    def increment_time_step(self):
        pass

    def reset_targets(self):
        pass

    def update_targets(self):
        pass

    @property
    def functional(self):
        pass

    
