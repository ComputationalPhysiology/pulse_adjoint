#!/usr/bin/env python
# Copyright (C) 2016 Henrik Finsberg
#
# This file is part of PULSE-ADJOINT.
#
# PULSE-ADJOINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PULSE-ADJOINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PULSE-ADJOINT. If not, see <http://www.gnu.org/licenses/>.
import os
from dolfin import *
from dolfin_adjoint import *


import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

import pulse_adjoint.models.material as mat
from pulse_adjoint.lvsolver import LVSolver
from pulse_adjoint.utils import Text, logger
from pulse_adjoint.dolfin_utils import (QuadratureSpace,
                                        MyReducedFunctional,
                                        RegionalParameter)
from pulse_adjoint.setup_optimization import (initialize_patient_data,
                                              check_patient_attributes,
                                              make_solver_params)
from pulse_adjoint.setup_parameters2 import (setup_general_parameters,
                                             setup_adjoint_contraction_parameters,
                                             setup_optimization_weigths)

from pulse_adjoint.numpy_mpi import*
from pulse_adjoint.optimization_targets import (OptimizationTarget,
                                                Regularization,
                                                RegionalStrainTarget,
                                                VolumeTarget)

from pulse_adjoint.forward_runner import PassiveForwardRunner, ActiveForwardRunner
from pulse_adjoint.config import PASSIVE_INFLATION_GROUP, PHASES
from pulse_adjoint.run_optimization import (solve_oc_problem,
                                            run_unloaded_optimization as unloaded)


from mesh_generation.mesh_utils import generate_fibers
from mesh_generation.idealized_geometry import mark_strain_regions
from mesh_generation.generate_mesh import setup_fiber_parameters
from mesh_generation.strain_regions import make_crl_basis

# This is temporary
foc = 1.54919333848


def setup_params(phase, space = "CG_1", mesh_type = "lv",
                 opt_targets=None, active_model = "active_strain"):
    setup_general_parameters()
    params = setup_adjoint_contraction_parameters()

    if opt_targets is not None:
        assert isinstance(opt_targets, dict)
        for key, val in opt_targets.items():
            params["Optimization_weigths"][key] = val
    # for key in params["Optimization_targets"].keys():
    #     if key in opt_targets:
    #         params["Optimization_targets"][key] = True
    #     else:
    #         params["Optimization_targets"][key] = False

    # Update weights
    # pparams = setup_passive_optimization_weigths()
    # aparams = setup_active_optimization_weigths()

    # params.remove('Passive_optimization_weigths')
    # params.add(pparams)
    # params.remove('Active_optimization_weigths')
    # params.add(aparams)

    # params['Active_optimization_weigths']["regularization"] = 0.1
    # params['Passive_optimization_weigths']["regularization"] = 0.1
    
    if phase == "passive":
        params["phase"] = "passive_inflation"
        params["matparams_space"] = space
        params["gamma_space"] = "R_0"

    else:
        params["phase"] = "active_contraction"
        params["matparams_space"] = "R_0"
        params["gamma_space"] = space

    if active_model == "active_strain":
        params["T_ref"] = 0.7
    else:
        params["T_ref"] = 200.0
        
    params["active_model"] = active_model
    params["adaptive_weights"] = False
        
    # params["Patient_parameters"]["mesh_type"] = mesh_type
    # params["Patient_parameters"]["patient"] = "mesh_simple_1"
    # params["Patient_parameters"]["patient_type"] = "test"

    
    params["sim_file"] = "test.h5"

    set_log_active(True)

    logger.setLevel(DEBUG)

    return params


def my_taylor_test(Jhat, m0_fun):

    
    m0 = gather_broadcast(m0_fun.vector().array())
      
    Jm0 = Jhat(m0)
    DJm0 = Jhat.derivative(forget=False)
    

    d = np.array([1.0]*len(m0)) #perturbation direction
    grad_errors = []
    no_grad_errors = []
   
    # epsilons = [0.05, 0.025, 0.0125]
    epsilons = [0.005, 0.0025, 0.00125]
        
    for eps in epsilons:
        m = np.array(m0 + eps*d)
        
        Jm = Jhat(m)
        no_grad_errors.append(abs(Jm - Jm0))
        grad_errors.append(abs(Jm - Jm0 - np.dot(DJm0, m - m0)))
       
    logger.info("Errors without gradient: {}".format(no_grad_errors))
    logger.info("Convergence orders without gradient (should be 1)")
    logger.info("{}".format(convergence_order(no_grad_errors)))
   
    logger.info("\nErrors with gradient: {}".format(grad_errors))
    logger.info("Convergence orders with gradient (should be 2)")
    con_ord = convergence_order(grad_errors)
    logger.info("{}".format(con_ord))
    
    assert (np.array(con_ord) > 1.85).all()


def store_results(params, rd, opt_result):
    from pulse_adjoint.run_optimization import store
    store(params, rd, opt_result)

    


def setup(phase, material_model, active_model,
          fiber_angle, weight, ndiv,
          control_regions, strain_regions, eps_vol,
          eps_strain, pressures, space, unload, h5name,
          geometry_index, isotropic, approx, restart):

    msg = ("\n\n"+" Test optimization on synthetic data  ".center(72, "#")+
           "\n\n\tphase = {}\n".format(phase) + \
           "\tmaterial_model = {}\n".format(material_model) + \
           "\tactive_model = {}\n".format(active_model) + \
           "\tfiber_angle (endo, epi) = {}\n".format(fiber_angle) + \
           "\tweights (strain, volume, regularization) = {}\n".format(weight) + \
           "\tcontrol_regions = {}\n".format(control_regions) + \
           "\tstrain_regions = {}\n".format(strain_regions)+ \
           "\teps (volume) = {}\n".format(eps_vol) + \
           "\teps (strain) = {}\n".format(eps_strain) + \
           "\tspace = {}\n".format(space) + \
           "\tunload = {}\n".format(unload) + \
           "\th5name = {}\n".format(h5name) + \
           "\tgeometry_index = {}\n".format(geometry_index) + \
           "\tisotrpoic = {}\n".format(isotropic) + \
           "\tapproximaion = {}\n\n".format(approx) + \
           "".center(72, "#") + "\n")
    print(msg)

    
    matparams = get_matparams(material_model, isotropic)
    expr = get_expr(phase)   
    ap_params = get_application_parameters(space, phase,
                                           active_model, weight,
                                           matparams, restart)
    ap_params["sim_file"] = h5name

    if os.path.isfile(h5name) and restart:
        os.remove(h5name)

        
    ap_params["volume_approx"] = approx
    ap_params["Patient_parameters"]["geometry_index"] = geometry_index
    
    
    fiber_params = get_fiber_params(fiber_angle)
    geo, basis, f0 = create_geometry(ndiv, fiber_params)
    control_markers, strain_markers = get_markers(geo.mesh, control_regions, strain_regions)
    
    geo.passive_filling_duration = len(pressures) if phase == "passive" else 1
    if unload:
        geo.passive_filling_duration -= 1
    
    # Make the solver parameters
    params, pressure_expr, control = make_solver_parameters(geo, ap_params)


    
    # params["passive_filling_duration"] = geo.passive_filling_duration
    params["basis"] = basis
    params["f0"] = f0
    params["strain_markers"] = strain_markers
    params["control_markers"] = control_markers
    params["nregions"] = sum(strain_regions)
    params["active_model"] = active_model
    params["material_model"] = material_model
    params["markers"] = geo.markers
    params["material_parameters"] = matparams
    params["dmu"] = Measure("dx", subdomain_data = strain_markers, domain = geo.mesh)
    params["dmu_control"] = Measure("dx", subdomain_data = control_markers, domain = geo.mesh)

    
    parameters["adjoint"]["stop_annotating"] = True
    strains, vols,  ws, us, ps, f_ex = generate_data(expr, params, ap_params,
                                                     pressure_expr, phase, eps_vol,
                                                     eps_strain, pressures)


    print("Exact control = {}\n".format(f_ex.vector().array()))
    
    if unload:
        params, ap_params, pressure_expr, basis, u_img = create_unloaded_geometry(params, us, ap_params,
                                                                         fiber_params,
                                                                         control_regions,
                                                                         strain_regions)
    else:
        u_img = None
            
    return params, strains, vols, ap_params, pressure_expr, ws, us, ps, f_ex, basis, u_img




def get_markers(mesh, control_regions, strain_regions):
    

    control_markers = mark_strain_regions(mesh, foc, control_regions, mark_mesh =False)
    strain_markers =  mark_strain_regions(mesh, foc, strain_regions, mark_mesh =False)

    return control_markers, strain_markers

def get_fiber_params(fiber_angle):
    fiber_params = setup_fiber_parameters()
    fiber_params["fiber_angle_endo"] = fiber_angle[0]
    fiber_params["fiber_angle_epi"] = fiber_angle[1]
    return fiber_params

def create_geometry(ndiv, fiber_params):

    from pulse_adjoint import LVTestPatient
    geo = LVTestPatient()
    mesh = geo.mesh

    
    e_circ = geo.circumferential
    e_long = geo.longitudinal
    e_rad = geo.radial

    basis = {"circumferential":e_circ, "longitudinal": e_long, "radial": e_rad}
    #fields = generate_fibers(mesh, fiber_params, geo.ffun)
    #f0 = fields[0]

    #geo.fiber = f0
    f0 = geo.fiber
    return geo, basis, f0

def create_unloaded_geometry(params, us, ap_params, fiber_params, control_regions, strain_regions):

    # Create new loaded geometry
    V = VectorFunctionSpace(params["mesh"], "CG", 1)
    geo_idx = int(ap_params["Patient_parameters"]["geometry_index"])
    if geo_idx >=0:
        geo_idx +=1
        
        
    if ap_params["volume_approx"] == "project":
        u_img = project(us[geo_idx], V)
    else:
        #interpolate
        u_img = interpolate(us[geo_idx], V)
        
    mesh_img = Mesh(params["mesh"])
    ALE.move(mesh_img, u_img)
    
    # local basis
    c,r,l = make_crl_basis(mesh_img, foc)
    basis = {"circumferential":c, "longitudinal": l, "radial": r}

    ffun_img = MeshFunction("size_t", mesh_img, 2)
    ffun_img.array()[:] = params["facet_function"].array()
    
    # fibers
    fields = generate_fibers(mesh_img, ffun = ffun_img, **fiber_params)
    f0 = fields[0]
    
    
    
    
    geo_img = lambda : None
    geo_img.mesh = mesh_img
    geo_img.ffun = ffun_img
    geo_img.markers = params["markers"]
    geo_img.fiber = f0

    N = FacetNormal(geo_img.mesh)
    X = SpatialCoordinate(geo_img.mesh)
    ds = Measure("exterior_facet",
                 subdomain_data = geo_img.ffun,
                 domain = geo_img.mesh)(30)
    vol = assemble((-1.0/3.0)*dot(X,N)*ds)
    print vol

    
    if os.path.isfile("synthetic_unloaded.h5"):
        os.remove("synthetic_unloaded.h5")
    
    from mesh_generation.mesh_utils import save_geometry_to_h5
    h5name = "synthetic_unloaded.h5"
    save_geometry_to_h5(mesh_img, h5name, "", params["markers"],
                        fields, [c,r,l], other_functions={"u_img":u_img})
    
    ap_params["Patient_parameters"]["mesh_group"] =""
    ap_params["Patient_parameters"]["mesh_path"] = h5name
    ap_params["passive_weights"] = "all"
    ap_params["unload"] = True
    ap_params["Unloading_parameters"]["tol"] = 1e-6
    ap_params["Unloading_parameters"]["maxiter"] = 40
    ap_params["Unloading_parameters"]["method"] = "hybrid"#"fixed_point"
    
    ap_params["Unloading_parameters"]["unload_options"]["maxiter"] = 40
    ap_params["Unloading_parameters"]["unload_options"]["tol"] = 1e-10
    ap_params["Optimization_parameters"]["passive_maxiter"] = 40
    
    control_markers, strain_markers = get_markers(mesh_img, control_regions, strain_regions)

    params, p_lv, control = make_solver_parameters(geo_img, ap_params)
    params["basis"] = basis
    params["f0"] = f0
    params["strain_markers"] = strain_markers
    params["control_markers"] = control_markers
    params["dmu"] = Measure("dx", subdomain_data = strain_markers, domain = mesh_img)
    params["dmu_control"] = Measure("dx", subdomain_data = control_markers, domain = mesh_img)
    parameters["adjoint"]["stop_annotating"] = False

    return params, ap_params, p_lv, basis, u_img


def get_matparams(material_model, isotropic = True):

    # Material coefficients
    if material_model == "neo_hookean":
        mu = 0.385
        matparams = {"mu": mu}
    else:
        a = 1.0
        b = 1.0
        a_f = 0.0 if isotropic else 1.0
        b_f = 1.0

        matparams = {"a":a, "b":b, "a_f":a_f, "b_f":b_f}

    return matparams
        

def get_expr(phase):
    
    if phase == "passive":
        expr = Expression("x[2]-x[1]+x[0]+10.0", degree=1)
        
    else:
        expr =  Expression("0.1*(1.0+0.3*(x[2]+x[1]+x[0]))", degree=1)

    return expr

def make_solver_parameters(geo, ap_params):

    
    check_patient_attributes(geo)
    solver_parameters, p_lv, control = make_solver_params(ap_params, geo)
    return solver_parameters, p_lv, control


def get_optimal_gamma(params, ap_params):


    if ap_params["gamma_space"] == "regional":
        paramvec = RegionalParameter(params["control_markers"])
    else:
        family, degree = ap_params["gamma_space"].split("_")                
        paramvec = dolfin.Function(dolfin.FunctionSpace(params["mesh"], family, int(degree)), name = "matparam vector")
    
    # gamma = Function(V_a)
    main_active_group = "active_contraction"
    passive_group = "passive_inflation"
    active_group = "/".join([main_active_group, "contract_point_0"])


    with HDF5File(mpi_comm_world(), ap_params["sim_file"], "r") as h5file:
        
        h5file.read(paramvec, "/".join([active_group.format(p), "optimal_control"]))

    return paramvec

def get_optimal_matparam(params, ap_params):

    if ap_params["matparams_space"] == "regional":
        paramvec = RegionalParameter(params["control_markers"])
    else:
        family, degree = ap_params["matparams_space"].split("_")                
        paramvec = dolfin.Function(dolfin.FunctionSpace(params["mesh"], family, int(degree)), name = "matparam vector")

   
    passive_group = "passive_inflation"
        
    with HDF5File(mpi_comm_world(), ap_params["sim_file"], "r") as h5file:
        
        h5file.read(paramvec,  "/".join([passive_group, "optimal_control"]))

    return paramvec

def get_regional_exr(expr, params):
    
    control_markers = params["control_markers"]
    a = RegionalParameter(control_markers)
    V = FunctionSpace(params["mesh"], "DG", 0)
    
    a_tmp = interpolate(expr, V)
    a_regional = []
    

        
    for region in set(control_markers):
        region_vol = assemble(Constant(1.0)*params["dmu_control"](int(region)))
        a_regional.append(1/float(region_vol) * assemble(a_tmp*params["dmu_control"](int(region))))
        
    # print a_regional
    # a.assign(Constant(a_regional))
    # a.vector()[:] = np.array(a_regional)
    # print a_regional
    assign_to_vector(a.vector(), np.array(a_regional))
    # exit()
    # assign_to_vector(a.vector(), gather_broadcast(np.array(a_regional)))
    # a.vector().set_local(np.array(a_regional))
    
    return a

def generate_data(expr, params, ap_params, pressure_expr, phase,
                  eps_vol = 0.0, eps_strain = 0.0, pressures = [0.3]):


    logger.info("\n\nGenerate synthetic data....\n")

    space_str = "matparams_space" if phase == "passive" \
                else "gamma_space"
    
    # Create an object for each single material parameter
    if ap_params[space_str] == "regional":
        
        f = get_regional_exr(expr, params)
        
    else:
        
        family, degree = ap_params[space_str].split("_")
        space = FunctionSpace(params["mesh"], family, int(degree))
      
        
        # The linear isotropic parameter
        f = interpolate(expr, space)

    if phase == "passive":
        params["material_parameters"]["a"] = f
        act = Constant(0.0)
    else:
        act = f
    
   
    if params["material_model"] == "neo_hookean":
        material = mat.NeoHookean(params["f0"], act,
                                  params["material_parameters"],
                                  active_model = ap_params["active_model"])
    else:
        material = mat.HolzapfelOgden(params["f0"], act,
                                      params["material_parameters"],
                                      active_model = ap_params["active_model"])

    params["material"] = material



    solver = LVSolver(params)
    solver.parameters["solve"]["newton_solver"]["report"] = True
    solver.solve()
    
    if phase == "active":
        h5group =  PASSIVE_INFLATION_GROUP
        with HDF5File(mpi_comm_world(), ap_params["sim_file"], "w") as h5file:
            h5file.write(solver.get_state(), "/".join([h5group, "states/0"]))


    us_vol = []
    us_strain = []
    ps = []
    ws = []

    V_cg1 = VectorFunctionSpace(params["mesh"], "CG", 1)
    for pres in pressures:


        pressure_expr["p_lv"].assign(pres)
        solver.solve()
        w = solver.get_state()
        u,p = w.split(deepcopy=True)


        if ap_params["volume_approx"] == "project":
            us_vol.append(project(u.copy(True), V_cg1))
        elif ap_params["volume_approx"] == "interpolate":
            us_vol.append(interpolate(u.copy(True), V_cg1))
        else:
            us_vol.append(u.copy(True))

        if ap_params["strain_approx"] == "project":
            us_strain.append(project(u.copy(True), V_cg1))
        elif ap_params["strain_approx"] == "interpolate":
            us_strain.append(interpolate(u.copy(True), V_cg1))
        else:
            us_strain.append(u.copy(True))
            
        ps.append(p.copy(True))
        ws.append(w.copy(True))
    if 0:
        if ap_params[space_str] == "regional":
            V = FunctionSpace(params["mesh"], "DG", 0)
            f_plot = project(f.get_function(), V)
        else:
            f_plot = f
    
        plot(f_plot, title = "a", mode = "color")
        plot(u, mode = "displacement", title = "u")
        interactive()
        exit()
    
    target_strain = RegionalStrainTarget(params["mesh"],
                                  params["basis"],
                                  params["dmu"],
                                  nregions = params["nregions"],
                                  tensor = "gradu")

    target_strain.set_target_functions()

    strains_arr = []
    vols = []
    logger.info("\n\t{:>10}\t{:>10}\t{:>10}".format("Pressure", "Volume", "Vol+noise"))
    for k, (u_vol, u_strain) in enumerate(zip(us_vol, us_strain)):

        if params["basis"]:
            # strains_zeros = []
            strains_arr_it = []
            target_strain.assign_simulated(u_strain)
            for i in range(target_strain.nregions):
                s = gather_broadcast(target_strain.simulated_fun[i].vector().array())
                
                noise = [0.0]*len(target_strain.crl_basis) if eps_strain <= 0 \
                        else np.random.normal(0,eps_strain, len(target_strain.crl_basis))
                s_tot = np.add(s, noise)

                strains_arr_it.append(s_tot)
            
            strains_arr.append(strains_arr_it)
        
        
        if params.has_key("markers"):
            dS = Measure("exterior_facet",
                         subdomain_data = params["facet_function"],
                         domain = params["mesh"])(params["markers"]["ENDO"][0])
        
            X = SpatialCoordinate(params["mesh"])
            N = FacetNormal(params["mesh"])
            vol_ref = assemble((-1.0/3.0)*dot(X, N)*dS)
            
            target_vol = VolumeTarget(params["mesh"], dS, "LV", "project")
            target_vol.set_target_functions()
            target_vol.assign_simulated(u_vol)
            vol = gather_broadcast(target_vol.simulated_fun.vector().array())[0]
            noise = 0.0 if eps_vol <= 0 else np.random.normal(0,eps_vol)
            vol_tot = vol+noise

            logger.info("\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}".format(pressures[k], vol, vol_tot))
            
            if len(vols) == 0:
                vol_prev = vol_ref
            else:
                vol_prev = vols[-1][-1]

            vols.append([vol_prev, vol_tot])

        else:
            vols = None

    logger.info("Done generating synthetic data.\n")
            
    return strains_arr, vols, ws, us_vol, ps, f
  

    



def get_application_parameters(space = "CG_1", phase="passive",
                               active_model = "active_strain",
                               weight=None, matparams = None, restart = True):
    ap_params = setup_adjoint_contraction_parameters()

    if weight is not None:
        ap_params["Passive_optimization_weigths"]["regional_strain"] = weight[0]
        ap_params["Passive_optimization_weigths"]["volume"] = weight[1]
        ap_params["Passive_optimization_weigths"]["regularization"] = weight[2]

        ap_params["Active_optimization_weigths"]["regional_strain"] = weight[0]
        ap_params["Active_optimization_weigths"]["volume"] = weight[1]
        ap_params["Active_optimization_weigths"]["regularization"] = weight[2]

    if matparams:
        for k, v in matparams.iteritems():
            ap_params["Material_parameters"][k] = v

    ap_params["Optimization_parameters"]["passive_maxiter"] = 100
    ap_params["Optimization_parameters"]["passive_opt_tol"] = 1e-10
    ap_params["Optimization_parameters"]["active_maxiter"] = 100
    ap_params["Optimization_parameters"]["active_opt_tol"] = 1e-16

    ap_params["adaptive_weights"] = False
    ap_params["active_relax"] = 1.0
    ap_params["passive_relax"] = 1.0

    
    if phase == "active":
        ap_params["phase"] = PHASES[1]
        
        ap_params["gamma_space"] = space
    else:
        ap_params["phase"] = PHASES[0]
        
        ap_params["matparams_space"] = space
        
    ap_params["passive_weights"] = "-1"
    
    
    ap_params["active_model"] = active_model
    ap_params["log_level"] = 20
    
    return ap_params
    
def run_optimization(params, strains, vols,  ap_params, pressures, p_lv, initial_guess = 1.0, return_rd = False):



    if ap_params["phase"] == PHASES[0]:
        space_str = "matparams_space"
    else:
        space_str = "gamma_space"
            
    if ap_params[space_str] == "regional":
        paramvec = RegionalParameter(params["control_markers"])
        
    else:
        
        family, degree = ap_params[space_str].split("_")
        space = FunctionSpace(params["mesh"], family, int(degree))
        paramvec = Function(space, name = "control")
        

    val = Constant(initial_guess) if paramvec.value_size() == 1 \
          else Constant([initial_guess]*paramvec.value_size())

    paramvec.assign(val)
    
    matparams = params["material_parameters"]

    if ap_params["phase"] == PHASES[0]:
        matparams["a"] = paramvec
        act = Constant(0.0)
        runner = PassiveForwardRunner
        relax = "passive_relax"
    else:
        act = paramvec
        runner = ActiveForwardRunner
        relax = "active_relax"

    # act = Constant(0.0)
    if params["material_model"] == "neo_hookean":
        material = mat.NeoHookean(params["f0"], act,
                                  matparams,
                                  active_model = params["active_model"])
    else:
        material = mat.HolzapfelOgden(params["f0"], act,
                                      matparams, 
                                      active_model = params["active_model"])
         
    params["material"] = material

    
   
    
    
    
    
    
    parameters["adjoint"]["stop_annotating"] = True
    opt_target_strain = RegionalStrainTarget(params["mesh"],
                                             params["basis"],
                                             params["dmu"],
                                             nregions = params["nregions"])
    

    strain_dict = {k+1:[] for k in range(params["nregions"])}
    for j, strain in enumerate(strains):
        for i, s in enumerate(strain, start = 1):
            
            strain_dict[i].append(s)

        opt_target_strain.load_target_data(strain_dict, j)
  

    reg = Regularization(params["mesh"],
                         ap_params["matparams_space"],
                         ap_params["Passive_optimization_weigths"]["regularization"],
                         regtype = "L2_grad", mshfun = params["control_markers"])
    

    if params.has_key("markers"):
        dS = Measure("exterior_facet",
                     subdomain_data = params["facet_function"],
                     domain = params["mesh"])(params["markers"]["ENDO"][0])
        opt_target_vol = VolumeTarget(params["mesh"], dS, "LV")


        for i,v in enumerate(vols):
            opt_target_vol.load_target_data(v, 1)
            
    
        targets = {"regularization": reg,
                   "volume":opt_target_vol,
                   "regional_strain":opt_target_strain}
    else:
        targets = {"regularization": reg,
                   "regional_strain":opt_target_strain}
        
    parameters["adjoint"]["stop_annotating"] = False

    

    
    for_run = runner(params,
                     {"p_lv":p_lv},
                     {"pressure":pressures},
                     targets,
                     ap_params,
                     paramvec)


    forward_result, _ = for_run(paramvec, False)

    weights = {}
    for k, v in for_run.opt_weights.iteritems():
        weights[k] = v/(10*forward_result["func_value"])
    for_run.opt_weights.update(**weights)
    # print("Update weights for functional")
    # for_run._print_functional()
    
    parameters["adjoint"]["stop_annotating"] = True
    rd = MyReducedFunctional(for_run, paramvec, relax = ap_params[relax])

    if return_rd:
        return rd, targets
    
    rd(paramvec)
    solve_oc_problem(ap_params, rd, paramvec)

    return rd, forward_result, targets







def run_unloaded_optimization(params, strains, vols, ap_params, pressures,
                              p_lv, return_rd = False):

   
    
    patient = set_patient_attributes(params, pressures, vols, strains,  True)
    
    unloaded(ap_params, patient)


    

def set_patient_attributes(params, pressures, vols, strains, unload):

    from pulse_adjoint import FullPatient
    patient = FullPatient(init=False)
    patient.mesh = params["mesh"]
    patient.ffun = params["facet_function"]
    patient.fiber = params["f0"]
    patient.sfun = params["strain_markers"]
    patient.markers = params["markers"]
    start_idx = 1 if unload else 0
    patient.pressure = pressures[start_idx:]
    patient.volume = np.array(vols).T[1][start_idx:]
    patient.strain =  {i+1:[] for i in range(len(strains[0]))}
    patient.passive_filling_duration = len(pressures)-1 if unload else len(pressure)
    patient._mesh_type = "lv"
    for s in strains[start_idx:]:
        for i,si in enumerate(s, start=1):
            patient.strain[i].append(si)

    return patient
