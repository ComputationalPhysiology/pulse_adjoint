#!/usr/bin/env python
"""
The functional you want to minimize consists of
different optimzation targets.

It may consist of a volume-target and a regional strain-target
in which you functional may take the following form::

    functional = a*volume_target_form + b*strain_target_form

with::

    volume_target = VolumeTarget()
    volume_target_form = volume_target.get_form()

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
import numpy as np
import dolfin
import dolfin_adjoint

try:
    from dolfin_adjoint import (Function, interpolate, Constant,
                                project, assemble)
except ImportError:
    from dolfin import (Function, interpolate, Constant,
                        project, assemble)

from pulse.dolfin_utils import list_sum, compute_meshvolume, map_displacement
from pulse import numpy_mpi
from . import config
from pulse import kinematics

from pulse.utils import make_logger, get_lv_marker

logger = make_logger(__name__, 10)

# from .utils import make_logger

# logger = make_logger(__name__, config.log_level)

__all__ = ["RegionalStrainTarget", "FullStrainTarget",
           "VolumeTarget", "Regularization"]


class OptimizationTarget(object):
    """Base class for optimization target
    """
    def __init__(self, mesh):
        """
        Initialize base class for optimization targets

        :param mesh: The underlying mesh
        :type mesh: :py:class:`dolfin.Mesh`

        """

        # A real space for projecting the functional
        self.realspace = dolfin.FunctionSpace(mesh, 'R', 0)

        # These spaces are only used if you want to project
        # or interpolate the displacement before assigning it
        # Space for interpolating the displacement if needed
        self._interpolation_space = dolfin.VectorFunctionSpace(mesh, 'CG', 1)
        # Displacement space
        self._disp_space = dolfin.VectorFunctionSpace(mesh, 'CG', 2)

        # The volume of the mesh
        self.meshvol = compute_meshvolume(mesh)
        # Test and trial functions for the target space
        self._trial = dolfin.TrialFunction(self.target_space)
        self._test = dolfin.TestFunction(self.target_space)

        # Test and trial functions for the real space
        self._trial_r = dolfin.TrialFunction(self.realspace)
        self._test_r = dolfin.TestFunction(self.realspace)

        # List for the target data
        self.data = []

        # List for saved data
        self.results = {"func_value": [],
                        "target": [],
                        "simulated": []}
        self.reset()
        self.set_target_functions()

    def reset(self):
        self.func_value = 0.0
        self.results["target"] = []
        self.results["simulated"] = []

    def save(self):
        self.func_value += self.get_value()
        self.results["func_value"].append(self.func_value)
        self.results["target"].append(dolfin.Vector(self.target_fun.vector()))
        self.results["simulated"]\
            .append(dolfin.Vector(self.simulated_fun.vector()))

    def next_target(self, it, annotate=False):
        self.assign_target(self.data[it], annotate)

    def set_target_functions(self):
        """Initialize the functions
        """

        self.target_fun \
            = dolfin_adjoint.Function(self.target_space,
                                      name="Target {}".format(self._name))
        self.simulated_fun \
            = dolfin_adjoint.Function(self.target_space,
                                      name="Simulated {}".format(self._name))
        self.functional = dolfin_adjoint.Function(self.realspace)

        self._set_form()

    def load_target_data(self, target_data, n):
        """Load the target data

        :param target_data: The data
        :param n: Index

        """
        f = dolfin.Function(self.target_space)
        numpy_mpi.assign_to_vector(f.vector(), np.array(target_data[n]))
        self.data.append(f)

    def _set_form(self):
        """The default form is just the least square
        difference
        """
        self._form = (self.target_fun - self.simulated_fun)**2

    def assign_target(self, target, annotate=False):
        """Assing target value to target function

        :param target: new target
        """
        logger.debug("Assign target for {}".format(self._name))
        self.target_fun.assign(target, annotate=annotate)

    def assign_functional(self):

        logger.debug("Assign functional for {}".format(self._name))
        dolfin_adjoint.solve(self._trial_r*self._test_r*dolfin.dx ==
                             self._test_r*self._form*dolfin.dx,
                             self.functional)

    def get_functional(self):
        """Return the integral form of the functional
        We devide by the volume, so that when integrated
        the value of the functional is the value of the
        integral.
        """
        return (self.functional / self.meshvol)

    def get_simulated(self):
        return self.simulated_fun

    def get_target(self):
        return self.target_fun

    def get_value(self):
        return numpy_mpi.gather_broadcast(self.functional.vector().array())[0]


class RegionalStrainTarget(OptimizationTarget):
    """Class for regional strain optimization
    target
    """
    def __init__(self, mesh, crl_basis, dmu, weights=None,
                 nregions=None,tensor="gradu",
                 F_ref=None, approx="original",
                 map_strain=False):
        """
        Initialize regional strain target

        Arguments
        ---------
        mesh: :py:class:`dolfin.Mesh`
            The mesh
        crl_basis: dict
            Basis function for the cicumferential, radial
            and longituginal components
        dmu: :py:class:`dolfin.Measure`
            Measure with subdomain information
        weights: :py:function:`numpy.array`
            Weights on the different segements
        nregions: int
            Number of strain regions
        tensor: str
            Which strain tensor to use, e.g gradu, E or almansi
        F_ref: :py:class:`dolfin.Function`
            Tensor to map strains to reference
        """
        self._name = "Regional Strain"

        assert tensor in ["gradu", "E", "almansi"]
        self._tensor = tensor
        self.approx = approx
        self._map_strain = map_strain
        if map_strain:
            from unloading.utils import normalize_vector_field

        dim = mesh.geometry().dim()
        self.dim = dim
        self._F_ref = F_ref if F_ref is not None else dolfin.Identity(dim)

        logger.debug("Load local basis.")
        logger.debug("Map local basis to new reference: {}".format(map_strain))
        self.crl_basis = []
        
        for l in ['c0', 'r0', 'l0']:
            msg = "{} : ".format(l)

            if hasattr(crl_basis, l):
                msg += "True"
                logger.debug(msg)
                basis_vector = getattr(crl_basis, l)
                if map_strain:

                    Fe = self._F_ref * basis_vector
                    logger.debug("Project")
                    e_ = dolfin.project(Fe)
                    logger.debug("Normalize")
                    e = normalize_vector_field(e_)
                else:
                    e = basis_vector

                self.crl_basis.append(e)

            else:
                msg += "False"
                logger.debug(msg)

        self.nbasis = len(self.crl_basis)

        msg = "Number of basis functions must be greater than zero"
        assert self.nbasis > 0, msg
        self.regions \
            = set(numpy_mpi.gather_broadcast(dmu.subdomain_data().array()))

        self.nregions = len(self.regions)
        if weights is None:
            self.weights_arr = np.ones((self.nregions, self.nbasis))
        else:
            self.weights_arr = weights

        self.target_space = dolfin.VectorFunctionSpace(mesh, "R", 0,
                                                       dim=self.nbasis)
        self.weight_space = dolfin.TensorFunctionSpace(mesh, "R", 0)
        self.dmu = dmu

        self.meshvols \
            = [Constant(dolfin.assemble(dolfin.Constant(1.0)
                                        * dmu(int(i)))) for i in self.regions]

        OptimizationTarget.__init__(self, mesh)

    def print_head(self):
        return "\t{:<10}".format("I_strain")

    def print_line(self):

        I = self.get_value()
        return "\t{:<10.2e}".format(I)

    def save(self):

        self.func_value += self.get_value()
        self.results["func_value"].append(self.func_value)
        target = []
        simulated = []
        for i in range(self.nregions):

            target.append(dolfin.Vector(self.target_fun[i].vector()))
            simulated.append(dolfin.Vector(self.simulated_fun[i].vector()))

        self.results["target"].append(target)
        self.results["simulated"].append(simulated)

    def load_target_data(self, target_data, n):
        """Load the target data

        :param dict target_data: The data
        :param int n: Index

        """
        strains = []
        for i in self.regions:
            # Do we need dolfin-adjoint here??
            f = dolfin.Function(self.target_space)
            if int(i) in target_data:
                numpy_mpi.assign_to_vector(f.vector(),
                                           np.array(target_data[int(i)][n]))
                strains.append(f)

        self.data.append(strains)

    def set_target_functions(self):
        """
        Initialize the functions

        """

        self.target_fun \
            = [dolfin_adjoint.Function(self.target_space,
                                       name="Target Strains_{}".format(i+1))
               for i in range(self.nregions)]

        self.simulated_fun \
            = [dolfin_adjoint.Function(self.target_space,
                                       name="Simulated Strains_{}".format(i+1))
               for i in range(self.nregions)]

        self.functional \
            = [dolfin_adjoint.Function(self.realspace)
               for i in range(self.nregions)]

        self.weights \
            = [dolfin_adjoint.Function(self.weight_space,
                                       name="Strains Weights_{}".format(i+1))
               for i in range(self.nregions)]

        self._set_weights()
        self._set_form()

    def _set_weights(self):

        for i in range(self.nregions):
            weight = np.zeros(self.nbasis**2)
            weight[0::(self.dim+1)] = self.weights_arr[i]
            numpy_mpi.assign_to_vector(self.weights[i].vector(), weight)

    def _set_form(self):

        self._form \
            = [(dolfin.dot(self.weights[i],
                           self.simulated_fun[i] - self.target_fun[i]))**2
               for i in range(self.nregions)]

    def get_value(self):
        return sum([numpy_mpi
                    .gather_broadcast(self.functional[i].vector().array())[0]
                    for i in range(self.nregions)])

    def assign_target(self, target, annotate=False):
        """Assing target regional strain

        :param target: Target regional strain
        :type target: list of :py:class:`dolfin.Function`
        """

        logger.debug("Assign target for {}".format(self._name))
        for fun, target in zip(self.target_fun, target):
            fun.assign(target, annotate=annotate)

    def assign_simulated(self, u):
        """Assing simulated regional strain

        :param u: New displacement
        :type u: :py:class:`dolfin.Function`
        """

        logger.debug("Assign simulated for {}".format(self._name))

        u_int = map_displacement(u, self._disp_space,
                                 self._interpolation_space,
                                 self.approx)
        # We need to correct for th reference deformation
        F = kinematics.DeformationGradient(u_int) \
            * dolfin.inv(self._F_ref)

        # Compute the strains
        if self._tensor == "gradu":
            tensor = kinematics.EngineeringStrain(F, isochoric=True)

        elif self._tensor == "E":
            tensor = kinematics.GreenLagrangeStrain(F, isochoric=True)

        elif self._tensor == "almansi":
            tensor = kinematics.EulerAlmansiStrain(F, isochoric=True)

        if len(self.crl_basis) > 0:

            tensor_diag \
                = dolfin.as_vector([dolfin.inner(tensor*e, e)
                                    for e in self.crl_basis])

            # Make a project for dolfin-adjoint recording
            for i, r in enumerate(self.regions):

                dolfin_adjoint.\
                    solve(dolfin.inner(self._trial,
                                       self._test) * self.dmu(int(r))
                          == dolfin.inner(self._test,
                                          tensor_diag) * self.dmu(int(r)),
                          self.simulated_fun[i],
                          solver_parameters={"linear_solver": "gmres"})
        else:

            logger.warning(('No local basis exist. '
                            'Regional strain cannot be computed'))

    def assign_functional(self):

        logger.debug("Assign functional for {}".format(self._name))
        for i, r in enumerate(self.regions):
            dolfin_adjoint.\
                solve(self._trial_r * self._test_r
                      / self.meshvol * dolfin.dx ==
                      self._test_r * self._form[i]
                      / self.meshvols[i] * self.dmu(int(r)),
                      self.functional[i])

    def get_functional(self):
        return (list_sum(self.functional) / self.meshvol)

                                          
class DisplacementTarget(OptimizationTarget):
    def __init__(self, mesh):
        self._name = "Displacement"
        self.dmu = dolfin.dx(mesh)
        self.target_space = dolfin.VectorFunctionSpace(mesh, "CG", 2)
        OptimizationTarget.__init__(self, mesh)

    def assign_simulated(self, u):
        """Assing simulated regional strain

        :param u: New displacement
        """

        # Make a project for dolfin-adjoint recording
        dolfin_adjoint.solve(dolfin.inner(self._trial,
                                          self._test) * self.dmu ==
                             dolfin.inner(u, self._test) * self.dmu,
                             self.simulated_fun)


class FullStrainTarget(OptimizationTarget):
    """Class for full strain field
    optimization target
    """
    def __init__(self, mesh, crl_basis):
        self._name = "Full Strain"
        self.dmu = dolfin.dx(mesh)
        self.crl_basis = crl_basis
        self.target_space = dolfin.VectorFunctionSpace(mesh, "CG", 1, dim=3)
        OptimizationTarget.__init__(self, mesh)

    def assign_simulated(self, u):
        """Assing simulated strain

        :param u: New displacement
        """

        # Compute the strains
        gradu = dolfin.grad(u)
        grad_u_diag = dolfin.as_vector([dolfin.inner(e, gradu*e)
                                        for e in self.crl_basis])

        # Make a project for dolfin-adjoint recording
        dolfin_adjoint.solve(dolfin.inner(self._trial, self._test)*self.dmu ==
                             dolfin.inner(grad_u_diag, self._test)*self.dmu,
                             self.simulated_fun)


class VolumeTarget(OptimizationTarget):
    """Class for volume optimization
    target
    """
    def __init__(self, mesh, dmu, chamber="LV", approx="project"):
        """Initialize the functions

        :param mesh: The mesh
        :param mesh: Surface measure of the endocardium
        
        """
        self._name = "{} Volume".format(chamber)
        self._X = dolfin.SpatialCoordinate(mesh)
        self._N = dolfin.FacetNormal(mesh)

        self.dmu = dmu
        self.chamber = chamber
        
        self.target_space = dolfin.FunctionSpace(mesh, "R", 0)
        self.endoarea = Constant(dolfin.assemble(dolfin.Constant(1.0)*dmu))

        assert approx in ["project", "interpolate", "original"]
        self.approx = approx
        OptimizationTarget.__init__(self, mesh)

    def print_head(self):
        return "\t{:<18}\t{:<20}\t{:<10}".format("Target {} Volume".format(self.chamber),
                                                 "Simulated {} Volume".format(self.chamber),
                                                 "I_{}".format(self.chamber))
    def print_line(self):
        v_sim = numpy_mpi\
            .gather_broadcast(self.simulated_fun.vector().array())[0]
        v_meas = numpy_mpi\
            .gather_broadcast(self.target_fun.vector().array())[0]
        I = self.get_value()
        
        return "\t{:<18.2f}\t{:<20.2f}\t{:<10.2e}".format(v_meas, v_sim, I)
        
    def load_target_data(self, target_data, n):
        """Load the target data

        :param target_data: The data
        :param n: Index

        """
        # Do we need dolfin-adjoint here?
        f = dolfin.Function(self.target_space)
        numpy_mpi.assign_to_vector(f.vector(), np.array([target_data[n]]))
        self.data.append(f)

    def assign_simulated(self, u):
        """Assign simulated volume

        :param u: New displacement
        """
        logger.debug("Assign simulated for {}".format(self._name))
        if u is None:

            vol = (-1.0/3.0) * dolfin.dot(self._X, self._N)

        else:
            u_int = map_displacement(u, self._disp_space,
                                     self._interpolation_space,
                                     self.approx)

            # Compute volume
            F = kinematics.DeformationGradient(u_int)
            J = kinematics.Jacobian(F)
            vol = (-1.0/3.0) * dolfin.dot(self._X + u_int,
                                          J * dolfin.inv(F).T * self._N)

        # Make a project for dolfin-adjoint recording
        dolfin_adjoint.solve(dolfin.inner(self._trial, self._test)
                             / self.endoarea * self.dmu ==
                             dolfin.inner(vol, self._test) * self.dmu,
                             self.simulated_fun)

    def _set_form(self):
        self._form = ((self.target_fun - self.simulated_fun)
                      / self.target_fun)**2


class Regularization(object):
    """Class for regularization
    of the control parameter
    """
    def __init__(self, mesh, spacestr="CG_1",
                 lmbda=0.0, regtype="L2_grad",
                 mshfun=None):
        """Initialize regularization object

        :param space: The mesh
        :param space: Space for the regularization
        :param lmbda: regularization parameter
        """
        # assert spacestr in ["CG_1", "regional", "R_0"], \
        #     "Unknown regularization space {}".format(space)

        self.spacestr = spacestr
        self.lmbda = lmbda
        self._value = 0.0

        self._mshfun = mshfun if mshfun is not None \
            else dolfin.MeshFunction("size_t", mesh,
                                     mesh.geometry().dim(),
                                     mesh.domains())

        self.meshvol = compute_meshvolume(mesh)

        self._regtype = regtype
        # A real space for projecting the functional
        self._realspace = dolfin.FunctionSpace(mesh, "R", 0)

        if spacestr == "regional":
            self._space = dolfin.FunctionSpace(mesh, "DG", 0)
        else:
            family, degree = spacestr.split("_")
            self._space = dolfin.FunctionSpace(mesh, family, int(degree))

        self.dx = dolfin.dx(mesh)
        self.results = {"func_value": []}
        self.reset()

    def print_head(self):
        return "\t{:<10}".format("I_reg")

    def print_line(self):
        I = self.get_value()
        return "\t{:<10.2e}".format(I)

    def reset(self):

        self.func_value = 0.0
        self._value = 0.0

    def save(self):

        self.func_value += self.get_value()
        self.results["func_value"].append(self.func_value)

    def set_target_functions(self):

        self.functional \
            = dolfin_adjoint.Function(self._realspace)

        if self.spacestr == "regional":
            from setup_optimization import RegionalParameter
            self._m = RegionalParameter(self._mshfun)
        else:
            self._m = dolfin.Function(self._space)

    def get_form(self):
        """Get the ufl form

        :returns: The functional form
        :rtype: (:py:class:`ufl.Form`)

        """

        if self._regtype == "L2":

            return (dolfin.inner(self._m, self._m)
                    / self.meshvol)*self.dx

        else:
            if self.spacestr == "CG_1":

                return (dolfin.inner(dolfin.grad(self._m),
                                     dolfin.grad(self._m))
                        / self.meshvol)*self.dx

            elif self.spacestr == "regional":

                expr_arr = ["0"] * self._m.value_size()

                # Sum all the components to find the mean
                expr_arr[0] = "1"
                m_sum = dolfin.dot(self._m,
                                   dolfin.Expression(tuple(expr_arr),
                                                     degree=1))
                expr_arr[0] = "0"

                for i in range(1, self._m.value_size()):
                    expr_arr[i] = "1"
                    m_sum += dolfin.dot(self._m,
                                        dolfin.Expression(tuple(expr_arr),
                                                          degree=1))
                    expr_arr[i] = "0"

                # Compute the mean
                m_avg = m_sum / self._m.value_size()

                # Compute the variance
                expr_arr[0] = "1"
                m_reg = (dolfin.dot(self._m,
                                    dolfin.Expression(tuple(expr_arr),
                                                      degree=1)) - m_avg)**2 \
                    / self._m.value_size()

                expr_arr[0] = "0"
                for i in range(1, self._m.value_size()):
                    expr_arr[i] = "1"
                    m_reg += (dolfin.dot(self._m,
                                         dolfin.Expression(tuple(expr_arr),
                                                           degree=1)) - m_avg)**2 \
                        / self._m.value_size()
                    expr_arr[i] = "0"

                # Create a functional term
                return (m_reg / self.meshvol) * self.dx

            else:
                return Constant(0.0)*self.dx

    def assign(self, m, annotate=False):
        self._m.assign(m, annotate=annotate)

    def get_functional(self):
        """Get the functional form
        (included regularization parameter)

        :param m: The function to be regularized
        :returns: The functional form
        :rtype: (:py:class:`ufl.Form`)

        """

        form = self.get_form()
        self._value = dolfin.assemble(form)

        return self.lmbda*form

    def get_value(self):
        """Get the value of the regularization term
        without regularization parameter

        :param m: The function to be regularized
        :returns: The value of the regularization term
        :rtype: float

        """
        return self._value


if __name__ == "__main__":

    from setup_parameters import setup_general_parameters
    setup_general_parameters()
    from mesh_generation import load_geometry_from_h5
    geo = load_geometry_from_h5("../tests/data/mesh_simple_1.h5")

   
    V = VectorFunctionSpace(geo.mesh, "CG", 2)
    u0 = Function(V)
    u1 = Function(V, "../tests/data/inflate_mesh_simple_1.xml")



    V0 = VectorFunctionSpace(geo.mesh, "CG", 1)
    
    
    
    dS = Measure("exterior_facet",
                 subdomain_data = geo.ffun,
                 domain = geo.mesh)(geo.markers["ENDO"][0])

    
    basis = {}
    for l in ["circumferential", "radial", "longitudinal"]:
        basis[l] = getattr(geo, l)
        
    dX = Measure("dx", subdomain_data=geo.sfun, domain=geo.mesh)
    nregions = len(set(geo.sfun.array()))


    for u in [u0, u1]:
        for approx in ["project", "interpolate", "original"]:


            # ui = u0
            ui = u1
        
            if approx == "interpolate":
                u_int = interpolate(project(ui, V),V0)
            
            elif approx == "project":
                u_int = project(ui, V0)

            else:
                u_int = ui





                
                       
            F_ref = grad(u_int) + Identity(3)
            

            target_vol = VolumeTarget(geo.mesh, dS, "LV", approx)
            target_vol.set_target_functions()
            target_vol.assign_simulated(u)
            
            vol = gather_broadcast(target_vol.simulated_fun.vector().array())[0]

            target_strain = RegionalStrainTarget(geo.mesh,
                                                 basis, 
                                                 dX,
                                                 nregions = nregions,
                                                 tensor = "gradu",
                                                 F_ref = F_ref,
                                                 approx=approx)

            target_strain.set_target_functions()
            target_strain.assign_simulated(u)

            strain = [gather_broadcast(target_strain.simulated_fun[i].vector().array()) \
                      for i in range(nregions)]
        
