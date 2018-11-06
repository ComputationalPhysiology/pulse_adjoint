#!/usr/bin/env python
"""
This script includes different functionlality that is needed
to compute the different features that we want to visualise.
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
from .args import *

def default_mechanical_features():
    from itertools import product
    return  [":".join(a) for a in product(["green_strain", "cauchy_stress",
                                           "cauchy_dev_stress"],
                                          ["longitudinal", "fiber",
                                           "circumferential", "radial"])] + \
                                           ["gamma:", "displacement:",
                                            "hydrostatic_pressure:",
                                            "I1:", "I1e:", "I4f:", "I4fe:"]

def asint(s):
    try: return int(s), ''
    except ValueError: return sys.maxint, s

def get_fiber_field(patient):

    if hasattr(patient, "e_f"):
        e_f = patient.e_f
    else:
        idx_arr = np.where([item.startswith("fiber") for item in dir(patient)])[0]
        
        if len(idx_arr) == 1:
            att = dir(patient)[idx_arr[0]]
            e_f = getattr(patient, att)
            
        else:
            raise ValueError("Unable to find fiber field")

    return e_f


def get_backward_displacement(patient, val):
    keys = val["unload"]["backward_displacement"].keys()
    key = sorted(keys, key = lambda t: int(t))[-1]

    u_arr = val["unload"]["backward_displacement"][key]

    if hasattr(patient, "original_geometry"):
        mesh = patient.original_geometry
    else:
        mesh = patient.mesh
        
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V)

    u.vector()[:] = u_arr
    
    return u

def get_sheets(patient, u):

    if patient.sheets:
        return patient.sheets

    # Otherwise we need to load the original sheets 
        
    from pulse_adjoint.unloading.utils import update_vector_field
    
    pass
    

def init_spaces(mesh, gamma_space = "CG_1"):

    from pulse_adjoint.utils import QuadratureSpace
    
    spaces = {}
    
    spaces["marker_space"] = dolfin.FunctionSpace(mesh, "DG", 0)
    spaces["stress_space"] = dolfin.FunctionSpace(mesh, "CG", 1)
    spaces["cg2"] = dolfin.FunctionSpace(mesh, "CG", 2)
    # spaces["cg3"] = dolfin.FunctionSpace(mesh, "CG", 3)
    # spaces["dg1"] = dolfin.FunctionSpace(mesh, "DG", 1)
    # spaces["dg2"] = dolfin.FunctionSpace(mesh, "DG", 2)
    # spaces["dg3"] = dolfin.FunctionSpace(mesh, "DG", 3)
    

    if gamma_space == "regional":
        spaces["gamma_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim = 17)
    else:
        gamma_family, gamma_degree = gamma_space.split("_")
        spaces["gamma_space"] = dolfin.FunctionSpace(mesh, gamma_family, int(gamma_degree))
        
    spaces["displacement_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    spaces["pressure_space"] = dolfin.FunctionSpace(mesh, "CG", 1)
    spaces["state_space"] = spaces["displacement_space"]*spaces["pressure_space"]
    spaces["strain_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=3)
    spaces["strainfield_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 1)

    
    # spaces["quad_space"] = QuadratureSpace(mesh, 4, dim = 1)
    
    return spaces

def compute_apical_registration(mesh, patient, endo_surf_apex):
    """
    Compute the displacement between the apex position 
    in the mesh and the apex position in the 
    segmented surfaces
    """

    ffun = patient.ffun
    ENDO = patient.ENDO
 
    endo_facets = np.where(ffun.array() == ENDO)
 
    endo_mesh_apex = [-np.inf, 0, 0]

    for f in dolfin.facets(mesh):
       
        if ffun[f] == ENDO:
            for v in dolfin.vertices(f):
                if v.point().x() > endo_mesh_apex[0]:
                    endo_mesh_apex = [v.point().x(),
                                      v.point().y(),
                                      v.point().z()]
   
    d_endo = np.subtract(endo_surf_apex, endo_mesh_apex)
   
    u = dolfin.Function(dolfin.VectorFunctionSpace(mesh, "CG", 1))
    u.assign(dolfin.Constant([d_endo[0], 0,0]))
    return u
   
   
def get_regional(dx, fun, fun_lst, regions = range(1,18), T_ref=1.0):
    """Return the average value of the function 
    in each segment

    :param dx: Volume measure marked according of AHA segments
    :param fun: The function that should be averaged
    :returns: The average value in each of the 17 AHA segment
    :rtype: list of floats

    """

    if fun.value_size() > 1:
        if len(fun_lst) == 1:
            return T_ref*fun_lst[0]
        else:
            return np.multiply(T_ref, fun_lst)

    meshvols = get_meshvols(dx, regions)
    lst = []
    for f in fun_lst:
        fun.vector()[:] = f
        lst_i = [] 
        for t, i in enumerate(regions):
            lst_i.append(T_ref*dolfin.assemble((fun/meshvols[t])*dx(i)))

        lst.append(lst_i)

    if len(fun_lst) == 1:
        return np.array(lst[0])


    return np.array(lst).T

def get_meshvols(dx, regions):

    meshvols = []
    for i in regions:
        meshvols.append(dolfin.Constant(dolfin.assemble(dolfin.Constant(1.0)*dx(i))))
    return meshvols

def get_regional_quad(dx, fun, regions):

    meshvols = get_meshvols(dx, regions)
    
    lst = []
    for i, r in enumerate(regions):

        lst.append(dolfin.assemble((fun/meshvols[i]) * dx(r)))
        
    return np.array(lst).T

def get_global_quad(dx, fun):

    meshvol = dolfin.assemble(dolfin.Constant(1.0)*dx)
    val = dolfin.assemble(fun * dx) / meshvol
        
    return val


    

def get_global(dx, fun, fun_lst, regions = range(1,18), T_ref = 1.0):
    """Get average value of function

    :param dx: Volume measure marked according of AHA segments
    :param fun: A dolfin function in the coorect spce
    :param fun_lst: A list of vectors that should be averaged
    :returns: list of average values (one for each element in fun_list)
    :rtype: list of floats

    """
    
    meshvols = []

    for i in regions:
        meshvols.append(dolfin.assemble(dolfin.Constant(1.0)*dx(i)))

    meshvol = np.sum(meshvols)

    fun_mean = []
    for f in fun_lst:
   
        fun.vector()[:] = f
        
        if fun.value_size() > 1:
            fun_tot = np.sum(np.multiply(fun.vector().array(), meshvols))
            
            
        else:
            fun_tot = 0
            for i in regions:            
                fun_tot += dolfin.assemble((fun)*dx(i))

        fun_mean.append(T_ref*fun_tot/meshvol)
 
    return fun_mean

def update_nested_dict(d,u):

    from collections import Mapping
    
    def update(d,u):
        for k, v in u.iteritems():
            if isinstance(v, Mapping):
                r = update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        return d

    update(d,u)

def recompute_strains_to_original_reference(strains, ref):

    strain_dict = {strain : {i:[] for i in STRAIN_REGION_NUMS}  for strain in STRAIN_NUM_TO_KEY.values()}
    
    for d in ['longitudinal', 'circumferential', 'radial']:
        for r in range(1,18):
            strain_trace = strains[d][r]
            new_strain_trace = np.zeros(len(strain_trace))
            ea0 = strain_trace[ref]
        
            for i in range(len(strain_trace)):
                                
                ei0 = strain_trace[i]
                eia = (ei0 - ea0)/(ea0 + 1)
                new_strain_trace[i] = eia

            new_strain_trace = np.roll(new_strain_trace, -ref)
            strain_dict[d][r] = new_strain_trace
            
    return strain_dict
def compute_inner_cavity_volume(mesh, ffun, marker, u=None, approx="project"):
    """
    Compute cavity volume using the divergence theorem. 

    :param mesh: The mesh
    :type mesh: :py:class:`dolfin.Mesh`
    :param ffun: Facet function
    :type ffun: :py:class:`dolfin.MeshFunction`
    :param int endo_lv_marker: The marker of en endocardium
    :param u: Displacement
    :type u: :py:class:`dolfin.Function`
    :returns vol: Volume of inner cavity
    :rtype: float

    """
    dS = dolfin.Measure("exterior_facet", subdomain_data=ffun, domain=mesh)(marker)
    from pulse_adjoint.optimization_targets import VolumeTarget
    target = VolumeTarget(mesh, dS, "LV", approx)
    target.set_target_functions()
    target.assign_simulated(u)
    return target.simulated_fun.vector().array()[0]



def get_volumes(disps, patient, chamber = "lv", approx="project"):


    if chamber == "lv":

        if patient.markers.has_key("ENDO"):
            marker = patient.markers["ENDO"][0]
        elif patient.markers.has_key("ENDO_LV"):
            marker = patient.markers["ENDO_LV"][0]
        else:
            raise ValueError

    else:
        
        assert chamber == "rv"

        if not patient.markers.has_key("ENDO_RV"):
            return []

        marker = patient.markers["ENDO_RV"][0]
        
        
    V = dolfin.VectorFunctionSpace(patient.mesh, "CG", 2)
    u = dolfin.Function(V)
   
 
    ffun = patient.ffun
    
    volumes = []
    if isinstance(disps, dict):
        times = sorted(disps.keys(), key=asint)
    else:
        times = range(len(disps))
    for t in times:
        us = disps[t]
        u.vector()[:] = us

      
        volumes.append(compute_inner_cavity_volume(patient.mesh, ffun,
                                                   marker, u, approx))

   
    return volumes

def get_regional_strains(disps, patient, unload=False,
                         strain_approx = "original",
                         strain_reference="0",
                         strain_tensor="gradu",
                         map_strain = False,
                         almansi=False,
                         *args, **kwargs):


    
    from pulse_adjoint.optimization_targets import RegionalStrainTarget
    dX = dolfin.Measure("dx",
                 subdomain_data = patient.sfun,
                        domain = patient.mesh)


    load_displacemet = (unload and not strain_reference== "unloaded") or \
                       (not unload and strain_reference == "ED")

    
    if load_displacemet:
    
        if strain_reference == "0":
            idx = 1
        else:
            #strain reference =  "ED"
            if unload:
                idx = patient.passive_filling_duration
            else:
                idx =  patient.passive_filling_duration-1

        u0 = dolfin.Function(dolfin.VectorFunctionSpace(patient.mesh,"CG", 2))
        if isinstance(disps, dict):
            u0.vector()[:] = disps[str(idx)]
        else:
            u0.vector()[:] = disps[idx]

        V = dolfin.VectorFunctionSpace(patient.mesh, "CG", 1)
        if strain_approx in ["project","interpolate"]:
            
            if strain_approx == "project":
                u0 = dolfin.project(u0, V)
            else:
                u0 = dolfin.interpolate(u0, V)

        else:
            u_int = dolfin.interpolate(u0, V)
                    
                
        F_ref = dolfin.grad(u0) + dolfin.Identity(3)
                

    else:
        F_ref = dolfin.Identity(3)

    if almansi:
        strain_tensor="almansi"


    crl_basis = {}
    basis_keys = []
    for att in ["circumferential", "radial", "longitudinal"]:
        if hasattr(patient, att):
            basis_keys.append(att)
            
            crl_basis[att] = getattr(patient, att)
            
    target = RegionalStrainTarget(patient.mesh,
                                  crl_basis, dX,
                                  F_ref =F_ref,
                                  approx = strain_approx,
                                  tensor = strain_tensor,
                                  map_strain=map_strain)
    target.set_target_functions()
   
    
    regions = target.regions

    strain_dict = {}
    
    for d in basis_keys:
        strain_dict[d] = {int(i):[] for i in regions}

    
    
   
    V = dolfin.VectorFunctionSpace(patient.mesh, "CG", 2)
    u = dolfin.Function(V)

    if isinstance(disps, dict):
        times = sorted(disps.keys(), key=asint)
    else:
        times = range(len(disps))
        
    for t in times:
        us = disps[t]
        u.vector()[:] = us

        target.assign_simulated(u)

        for i,d in enumerate(basis_keys):
            for j, r in enumerate(regions):
                strain_dict[d][r].append(target.simulated_fun[j].vector().array()[i])
                

    # error = np.sum([np.subtract(patient.strain[i].T[0],strain_dict["circumferential"][i][1:])**2 for i in range(3)], 0)
    # print error
    # exit()
    return strain_dict

def compute_strain_components(u, sfun, crl_basis, region, F_ref = dolfin.Identity(3), tensor_str="gradu"):

    mesh = sfun.mesh()
    dmu = dolfin.Measure("dx",
                         subdomain_data = sfun,
                         domain = mesh)
 
    # Strain tensor
    I = dolfin.Identity(3)
    F = (dolfin.grad(u) + dolfin.Identity(3))*dolfin.inv(F_ref)
    
    if tensor_str == "gradu":
        tensor = F-I
    else:
        C = F.T * F
        tensor = 0.5*(C-I)

    # Volume of region
    vol = dolfin.assemble(dolfin.Constant(1.0)*dmu(region))

    # Strain components
    return [dolfin.assemble(dolfin.inner(e,tensor*e)*dmu(region))/vol for e in crl_basis]

def interpolate_arr(x, arr, N, period = None, normalize = True):

    # from scipy.interpolate import splrep
    a_min = np.min(arr)
    a_max = np.max(arr)
    x_min = np.min(x)
    x_max = np.max(x)

    if a_min == a_max:
        # The array is constant
        return a_min*np.ones(N)
        
    # x = np.linspace(0,1,len(arr))

    # Normalize
    if normalize:
        arr = np.subtract(arr,a_min)
        arr = np.divide(arr, a_max-a_min)
        x = np.subtract(x, x_min)
        x = np.divide(x, x_max-x_min)

    # Interpolate
    xp = np.linspace(0,1,N)
    # try:
    fp = np.interp(xp,x,arr, period=period)
    # except:

    
    fp = np.multiply(fp, a_max-a_min)
    fp = np.add(fp, a_min)

    return fp

def interpolate_trace_to_valve_times(arr, valve_times, N):
    """
    Given an array and valvular timings, perform interpolation
    so that the resulting array is splitted up into chunks of length
    N, each chuck being the interpolated values of the array for 
    one valvular event to the next. 

    First list is from 'mvc' to 'avo'.
    Second list is from 'avo' to 'avc'.
    Third list is from 'avc' to 'mvo'.
    Fourth list is from 'mvo' to 'mvc'.

    :param arr: The array of interest 
    :type arr: `numpy.array` or list 
    :param dict valve_times: A dictionary of valvular timings
    :param int N: length of chunks
    :returns: a list of length 4, with each element being a
              list of length N. 
    :rtype: list

    """
    

    echo_valve_times = valve_times
    # The index when the given array is starting
    pfb = valve_times["passive_filling_begins"]
    
    n = len(arr)
    # Just some increasing sequence
    time = np.linspace(0,1, len(arr))
    

    # Roll array so that it start on the same index and in the valvular times
    arr_shift_pdb = np.roll(arr, pfb)
    # gamma_mean_shift = gamma_mean

    full_arr = []

    N_ = {"avo": int(3*N*float(0.05)), "avc":int(3*N*float(0.35)),
          "mvo":int(3*N*float(0.10)), "end":int(3*N*float(0.50)) }
    
    for start, end in [("mvc", "avo"), ("avo", "avc"), ("avc", "mvo"), ("mvo", "end")]:
        
        start_idx = echo_valve_times[start]
        end_idx = echo_valve_times[end]
        diff = (end_idx - start_idx) % n

        # If start and end are the same, include the previous point
        if diff == 0:
            start_idx -= 1
            diff = 1
        # if end == "mvc":
        #     diff -= 0

        # Roll array to this start
        arr_shift_start = np.roll(arr_shift_pdb, -start_idx)
        arr_partly = arr_shift_start[:diff+1]
        
        # The time starts at mvc
        time_shift_start = np.roll(arr_shift_pdb, echo_valve_times["mvc"]-start_idx)
        t_partly = time_shift_start[:diff+1]

        # just some increasing sequence
        dtime = time[:diff+1]
                
            
        darr_int = interpolate_arr(dtime, arr_partly, N_[end])
   
        
        full_arr.append(darr_int)
 
    return np.concatenate(full_arr)
def compute_elastance(state, pressure, gamma, patient,
                      params, matparams, return_v0 = False, chamber = "lv"):
    """FIXME! briefly describe function

    :param state: 
    :param pressure: 
    :param gamma: 
    :param gamma_space_str: 
    :param patient: 
    :param active_model: 
    :param matparams: 
    :param return_v0: 
    :returns: 
    :rtype: 

    """
    

    solver, p_expr = get_calibrated_solver(state, pressure,
                                         gamma, patient,
                                         params,matparams)

    u,_ = dolfin.split(solver.get_state())

    if patient.is_biv():
        p_expr["p_lv"].assign(pressure[0])
        p_expr["p_rv"].assign(pressure[1])
        
    else:
        p_expr["p_lv"].assign(pressure)


    solver.solve()

    assert chamber in ["lv", "rv"]
    if chamber == "lv":
        P = p_expr["p_lv"]
        if patient.markers.has_key("ENDO_LV"):
            endo_marker =  patient.markers["ENDO_LV"][0]
            pressure_ = pressure[0]
        else:
            endo_marker = patient.markers["ENDO"][0]
            pressure_ = pressure
    else:
        
        P = p_expr["p_rv"]
        endo_marker = patient.markers["ENDO_RV"][0]
        pressure_ = pressure[1]
    
    volume = compute_inner_cavity_volume(patient.mesh, patient.ffun,
                                         endo_marker, u)

    
    vs = [volume]
    ps = [pressure_]

    print "Original"
    print "{:10}\t{:10}".format("pressure", "volume")
    print "{:10.2f}\t{:10.2f}".format(pressure_, volume)
    print "Increase the pressure"

    n = 1
    inc = 0.1
    crash = True
    while crash:
        # Increase the pressure
        P_ = float(P) + inc
        P.assign(P_)
        # Do a new solve
        try:
            solver.solve()
        except SolverDidNotConverge:
            inc /= 2.0
            continue

        else:
            # Compute the new volume
            u,_ = dolfin.split(solver.get_state())
            v = compute_inner_cavity_volume(patient.mesh, patient.ffun,
                                            endo_marker, u)
            
            print "{:10.2f}\t{:10.2f}".format(float(P), v)
            # Append to the list
            vs.append(v)
            ps.append(float(P))

            crash = False

    if return_v0:
        e = np.mean(np.divide(np.diff(ps), np.diff(vs)))
        v0 = volume - float(pressure_)/e
        return e, v0
    else:
        return np.mean(np.divide(np.diff(ps), np.diff(vs)))


def compute_geometric_distance(patient, us, vtk_output):
    """Compute the distance between the vertices from the simulation
    and the vertices from the segmented surfaces of the endocardium.
    For each vertex in the simulated surface :math:`a \in \Xi_{\mathrm{sim}}`,
    we define the following distance measure

    .. math::

       d(a,\Xi_{\mathrm{seg}}) = \min_{b \in \Xi_{\mathrm{seg}}} \| a - b \| 

    where 

    .. math:: 

       \Xi_{\mathrm{seg}} 

    is the vertices of the (refined) segmented surface

    :param patient: Patient class
    :param us: list of displacemets
    :param vtk_output: directory were to save the output
    :returns: 
    :rtype: 

    """
    
    import vtk_utils
    import vtk
    
    
    V_cg1 = dolfin.VectorFunctionSpace(patient.mesh, "CG", 1)
    V_cg2 = dolfin.VectorFunctionSpace(patient.mesh, "CG", 2)
    u_current = dolfin.Function(V_cg2)
    u_prev = dolfin.Function(V_cg2)
    d = dolfin.Function(V_cg2)

    mean_dist = []
    max_dist = []
    std_dist = []
   

    for k,t in enumerate(np.roll(range(patient.num_points), -patient.passive_filling_begins)):

        mesh = patient.mesh

        if not us.has_key(str(k)):
            print("Time point {} does not exist".format(k))
            continue
        u_current.vector()[:] = us[str(k)]
        d.vector()[:] =  u_current.vector()[:] - u_prev.vector()[:]
        ud = dolfin.interpolate(d, V_cg1)
        dolfin.ALE.move(mesh, ud)
       
        
        endoname = vtk_utils.save_surface_to_dolfinxml(patient,t, vtk_output)
        endo_surf = dolfin.Mesh(endoname)
        endo_surf_apex = endo_surf.coordinates().T[0].max()
            
        # Registrer the apex
        u_apical = compute_apical_registration(mesh, patient, endo_surf_apex)
        dolfin.ALE.move(mesh, u_apical)
        

        # Save unrefined surface for later visualization
        surf_unrefined = vtk_utils.dolfin2polydata(endo_surf)
        distname = "/".join([vtk_output, "echopac_{}.vtk".format(k)])
        vtk_utils.write_to_polydata(distname, surf_unrefined)

        # Convert surface to dolfin format
        # surf_unrefined = dolfin2vtu(endo_surf)

        # Refine surface for better accuracy
        endo_surf_refined = dolfin.refine(dolfin.refine(dolfin.refine(dolfin.refine(endo_surf))))
        # Get endocardial mesh from original mesh
        endo_submesh = vtk_utils.get_submesh(mesh, patient.ENDO)

        # Convert surfaces to polydata
        endo_surf_vtk = vtk_utils.dolfin2polydata(endo_surf_refined)
        endo_submesh_vtk = vtk_utils.dolfin2polydata(endo_submesh)
        
        # Build a Kd search tree 
        tree = vtk.vtkKdTreePointLocator()
        tree.SetDataSet(endo_surf_vtk)
        tree.BuildLocator()

        distance = vtk.vtkDoubleArray()
        distance_arr = []
        for i in range(endo_submesh_vtk.GetNumberOfPoints()):
            p = endo_submesh_vtk.GetPoint(i)

            # Nearest neighbor
            idx = tree.FindClosestPoint(p)
            psurf = endo_surf_vtk.GetPoint(idx)

            # Compute di
            dist = np.linalg.norm(np.subtract(psurf,p))
            
            distance.InsertNextValue(dist)
            distance_arr.append(dist)

        # Set the distances as scalars in the vtk file
        endo_submesh_vtk.GetPointData().SetScalars(distance)

        distname = "/".join([vtk_output, "dist_{}.vtk".format(k)])
        vtk_utils.write_to_polydata(distname, endo_submesh_vtk)

        mean_dist.append(np.mean(distance_arr))
        std_dist.append(np.std(distance_arr))
        max_dist.append(np.max(distance_arr))

        u_prev.assign(u_current)

    d = {"mean_distance": mean_dist,
         "std_distance": std_dist,
         "max_distance": max_dist}
    return d

def get_Ivol(simulated, measured):
    """
    return the relatve error in l1 norm
    || V* - V ||_l1 / || V ||_l1 where V* is
    simulated volume and V is measured volume
    """
    if not len(simulated) == len(measured):
        print("All simulation points are not available")
        n = len(simulated)
        measured = measured[:n]
        
    return np.sum(np.abs(np.subtract(simulated,measured))) / \
        float(np.sum(measured))

def get_Istrain(simulated,measured):
    """
    Return two different measures for the strain error
    
    max:
    ||e* - e ||
    """
    I_strain_tot_rel = 0
    I_strain_tot_max = 0
    for d in measured.keys():
        
        I_strain_region_rel = []
        I_strain_region_max = []
        
        s_max = np.max([np.max(np.abs(s)) for s in measured[d].values()])
        for region in measured[d].keys():
            
            s_meas = measured[d][region]
            s_sim =  simulated[d][region]
            
            if not np.all(s_meas == 0):

                if not len(s_meas) == len(s_sim):
                    print("All simulation points are not available")
                    n = len(s_sim)
                    s_meas = s_meas[:n]
                    
                err_max =  np.divide(np.mean(np.abs(np.subtract(s_sim,s_meas))),
                                    s_max)
                err_rel = np.divide(np.sum(np.abs(np.subtract(s_sim,s_meas))),
                                    np.sum(np.abs(s_meas)))
                
                I_strain_region_max.append(err_max)
                I_strain_region_rel.append(err_rel)
  
        I_strain_tot_rel += np.mean(I_strain_region_rel)
        I_strain_tot_max += np.mean(I_strain_region_max)
                
    I_strain_rel = I_strain_tot_rel/3.
    I_strain_max = I_strain_tot_max/3.

    return I_strain_rel, I_strain_max

def copmute_data_mismatch(us, patient, measured_volumes, measured_strains):

    simulated_volumes = get_volumes(us, patient)
    simulated_strains = get_regional_strains(us, patient)
        
    I_vol = get_Ivol(simulated_volumes, measured_volumes)
    I_strain_rel, I_strain_max = get_Istrain(simulated_strains,
                                             measured_strains)

    data = {"I_strain_rel": I_strain_rel,
            "I_strain_max": I_strain_max,
            "I_vol": I_vol}

    return data

def compute_time_varying_elastance(patient, params, data):
    """Compute the elastance for every point in
    the cycle.

    :param patient: Patient class
    :param matparams: Optimal material parameters
    :param params: pulse_adjoint.adjoint_contraction_parameters
    :param val: data
    :returns: time varying elastance
    :rtype: list

    """

    
    matparams = {k:v[0] for k,v in data["material_parameters"].iteritems()}
    
    elastance = []
    dead_volume = []

    num_points = patient.num_points
    if params["unload"]: num_points += 1
    start = 1 if params["unload"] else 0
    
    for i in range(start, num_points):
        print "{} / {} ".format(i, num_points)
        
        p = patient.pressure[i]
        w = data["states"][str(i)]
        g = data["gammas"][str(i)]
        
        e, v0 = compute_elastance(w, p, g, patient, params,
                                  matparams, return_v0 = True)
        
        print "E = {}, V0 = {}".format(e, v0)
        elastance.append(e)
        dead_volume.append(v0)

    d = {"elastance": elastance, "v0":dead_volume}
    return d
    
    


def compute_cardiac_work_echo(stresses, strains, flip =False):
    """FIXME! briefly describe function

    :param list stresses: list of stresses
    :param list strains: list of strains
    :param bool flip: If true, change the sign on the stresses.
                      This is done in the original paper, when the
                      pressure plays the role as stress.
    :returns: the work
    :rtype: list

    """
    

    msg =  "Stresses and strains do not have same lenght"
    assert len(stresses) == len(strains), msg

    # Compute the averge
    stress_avg = np.add(stresses[:-1], stresses[1:])/2.0
    
    if flip:
        # Compute the shortening_rate
        dstrain = -np.diff(strains)
    else:
        # Compute the strain_rate
        dstrain = np.diff(strains)

    # The work is the cumulative sum of the product
    work = np.append(0,np.cumsum(dstrain*stress_avg))
    
    return work
    
    
    

def compute_cardiac_work(patient, params, val, case, wp, e_k = None):
    """Compute cardiac work. 

    :param patient: patient data
    :param params: pulse_adjoin.adjoint_contraction_parameters
    :param val: the data
    :param path: path to where to save the output

    """
    
    from cardiac_work import CardiacWork, CardiacWorkEcho, StrainEnergy


    spaces = get_feature_spaces(patient.mesh, params["gamma_space"])

    pressures = patient.pressure
    matparams = {k:v[0] for k,v in val["material_parameters"].iteritems()}

    states = val["states"]
    gammas = val["gammas"]
    times = sorted(states.keys(), key=asint)

    if params["unload"]:
        times = times[1:]
    

    dX = dolfin.Measure("dx",subdomain_data = patient.sfun,
                        domain = patient.mesh)
    
    V = dolfin.TensorFunctionSpace(patient.mesh, "DG", 1)
    W = dolfin.FunctionSpace(patient.mesh, "DG", 1)
    e_f = get_fiber_field(patient)
    

    
    # assert case in cases, "Unknown case {}".format(case)
    assert wp in work_pairs, "Unknown work pair {}".format(wp)

    reults = {}

    header = ("\nComputing Cardiac Work, W = {}\n"
              "{}, region = {}\n")

   
    if wp == "pgradu":
        cw = CardiacWorkEcho(V, W)
    elif wp == "strain_energy":
        cw = StrainEnergy()
    else:
        cw = CardiacWork(V, W)
            

    case_split = case.split("_")

                        
    if e_k is None:
        
        if len(case_split) == 1:
            e_k = None
        
        elif case_split[1] == "fiber":
            e_k = e_f

        elif case_split[1] == "sheet":


            e_k = get_sheets(patient)

        elif case_split[1] == "crosssheet":
            e_k = get_cross_sheets(patient)
        else:
            e_k = patient.longitudinal

                
    case_ = case_split[0]
        
    results = {}

    

    cw.reset()

    regions = [0]+ list(set(patient.sfun.array()))
    work_lst = {r:[] for r in regions}
    power_lst = {r:[] for r in regions}

    # print(header.format(wp, case, region))

    first_time = True

    # FIXME : Assume for now that we unload and the
    # strain should be computed with respect to
    # first point as reference.
    #{
    solver, p_lv = get_calibrated_solver(states["1"],
                                         pressures[1],
                                         gammas["1"],
                                         patient,
                                         params, 
                                         matparams)
        
    u_ref, _ = solver.get_state().split(deepcopy=True)
    I = dolfin.Identity(3)
    F_ref = dolfin.grad(u_ref) + I
    #}

   
    for t in times:

        print "\nTime: {}".format(t)
        state = states[t]
        gamma = gammas[t]
        pressure = pressures[int(t)]
        
        solver, p_lv = get_calibrated_solver(state, pressure,
                                             gamma,
                                             patient,
                                             params, 
                                             matparams)
        
        u,_ = solver.get_state().split(deepcopy=True)
        
        post = solver.postprocess()
            
        # Second Piola stress
        S = -post.second_piola_stress(deviatoric=False)
        Sdev = -post.second_piola_stress(deviatoric=True)
        # Green-Lagrange strain
        E = post.GreenLagrange()
        
        # # First Piola stress
        # P = solver.postprocess().first_piola_stress()
        # # Deformation gradient
        # F = post.deformation_gradient()

        # Strain energy
        psi = solver.postprocess().strain_energy()

        F_ = dolfin.grad(u) + I
        F = F_*dolfin.inv(F_ref)
        gradu = F - I
        
        if wp == "strain_energy":
            
            cw(psi, dx)
            
        else:
            if wp == "SE":
                stress = S
                strain = E
            elif wp == "SEdev":
                stress = Sdev
                strain = E
            # elif wp == "PF":
            #     stress = P 
            #     strain = F
            else:# wp == pgradu
                stress = pressure
                strain = gradu
                
                



        
        
        cw(strain, stress, case_, e_k)

        if first_time:
            first_time = False
            continue

        for region in regions:
            dx = dX if region == 0 else dX(int(region))
            meshvol = dolfin.assemble(dolfin.Constant(1.0)*dx)

            power = cw.get_power()
            work = cw.get_work()

            power_ = dolfin.assemble( power * dx ) / meshvol
            work_ = dolfin.assemble( work * dx ) / meshvol

            work_lst[region].append(work_)
            power_lst[region].append(power_)

            print("\t{:<10}\t{:<10.3f}\t{:<10.3f}".format(region, power_, work_))
        

    for region in regions:    
        results["{}_{}_region_{}".format(wp, case, region)] =  {"power":power_lst[region],
                                                                "work":work_lst[region]}
      

    return results
        
    
def get_feature_spaces(mesh, gamma_space = "CG_1"):

    spaces = {}

    spaces["marker_space"] = dolfin.FunctionSpace(mesh, "DG", 0)
    spaces["stress_space"] = dolfin.FunctionSpace(mesh, "CG", 1)
    spaces["cg1"] = dolfin.FunctionSpace(mesh, "CG", 1)
    spaces["cg2"] = dolfin.FunctionSpace(mesh, "CG", 2)
    # spaces["cg3"] = dolfin.FunctionSpace(mesh, "CG", 3)
    # spaces["dg1"] = dolfin.FunctionSpace(mesh, "DG", 1)
    # spaces["dg2"] = dolfin.FunctionSpace(mesh, "DG", 2)
    # spaces["dg3"] = dolfin.FunctionSpace(mesh, "DG", 3)

    if gamma_space == "regional":
        spaces["gamma_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim = 17)
    else:
        gamma_family, gamma_degree = gamma_space.split("_")
        spaces["gamma_space"] = dolfin.FunctionSpace(mesh, gamma_family, int(gamma_degree))
        
    spaces["displacement_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    spaces["pressure_space"] = dolfin.FunctionSpace(mesh, "CG", 1)
    # spaces["state_space"] = spaces["displacement_space"]*spaces["pressure_space"]
    # spaces["strain_space"] = dolfin.VectorFunctionSpace(mesh, "R", 0, dim=3)
    # spaces["strainfield_space"] = dolfin.VectorFunctionSpace(mesh, "CG", 1)

    from pulse_adjoint.utils import QuadratureSpace
    # spaces["quad_space"] = QuadratureSpace(mesh, 4, dim = 3)
    # spaces["quad_space_1"] = QuadratureSpace(mesh, 4, dim = 1)
    

    return spaces


def make_simulation(params, features, outdir, patient, data):

   

    if not features: return

    import vtk_utils
    

    # Mesh
    mesh = patient.mesh

    if 0:
        name = params["Patient_parameters"]["patient"]
        fname = "../DATA2/transformation/{}.txt".format(name)
        F = np.loadtxt(fname)

    else:
        F = np.eye(4)
        
    # Mesh that we move
    moving_mesh = dolfin.Mesh(mesh)


    # The time stamps
    if isinstance(data["gammas"], dict):
        times = sorted(data["gammas"].keys(), key=asint)
    else:
        times = range(len(data["gammas"]))

    if not hasattr(patient, "time"):
        patient.time = range(patient.num_points)
        
    time_stamps = np.roll(patient.time, -np.argmin(patient.time))
    from scipy.interpolate import InterpolatedUnivariateSpline
    s = InterpolatedUnivariateSpline(range(len(time_stamps)), time_stamps, k = 1)
    time_stamps = s(np.array(times, dtype=float))
    
    # Create function spaces
    spaces = get_feature_spaces(mesh, params["gamma_space"])
    moving_spaces = get_feature_spaces(moving_mesh, params["gamma_space"])
    if params["gamma_space"] == "regional":
        gamma_space = dolfin.FunctionSpace(moving_mesh, "DG", 0)
        sfun = merge_control(patient, params["merge_active_control"])
        rg = RegionalParameter(sfun)
    else:
        gamma_space = moving_spaces["gamma_space"]

    # Create functions

    # Markers
    sm = dolfin.Function(moving_spaces["marker_space"], name = "AHA-zones")
    sm.vector()[:] = patient.sfun.array()

    if hasattr(params["Material_parameters"]["a"], "vector"):
        matvec = params["Material_parameters"]["a"].vector()
    else:
        matvec = params["Material_parameters"]["a"]
    # Material parameter
    if params["matparams_space"] == "regional":
        mat_space = dolfin.FunctionSpace(moving_mesh, "DG", 0)
        sfun = merge_control(patient, params["merge_passive_control"])
        rmat = RegionalParameter(sfun)
        rmat.vector()[:] = matvec
        mat = dolfin.Function(mat_space, name = "material_parameter_a")
        m =  dolfin.project(rmat.get_function(), mat_space)
        mat.vector()[:] = m.vector()
        
    else:
        family, degree = params["matparams_space"].split("_")
        mat_space= dolfin.FunctionSpace(moving_mesh, family, int(degree))
        mat = dolfin.Function(mat_space, name = "material_parameter_a")
        mat.vector()[:] = matvec


    functions = {}
    functions_ = {}    
    for f in features.keys()+["gamma"]:

        if f == "displacement":
            pass
        elif f == "gamma":
            functions[f] = dolfin.Function(gamma_space, name="gamma")

        elif f == "hydrostatic_pressure":
            functions[f] = dolfin.Function(moving_spaces["pressure_space"], 
                                           name=f)
            functions_[f] = dolfin.Function(moving_spaces["pressure_space"], 
                                            name=f)
        
        else:
            functions[f] = dolfin.Function(moving_spaces["cg1"], 
                                           name=f)
            functions_[f] = dolfin.Function(moving_spaces["cg2"], 
                                            name=f)


    # Setup moving mesh
    u = dolfin.Function(spaces["displacement_space"])
    u_prev = dolfin.Function(spaces["displacement_space"])
    u_diff = dolfin.Function(spaces["displacement_space"])
    # Space for interpolation
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    # fiber = dolfin.Function(moving_spaces["quad_space"])
   
   
    fname = "simulation_{}.vtu"
    vtu_path = "/".join([outdir, fname])

    old_coords = np.ones((moving_mesh.coordinates().shape[0], 4))
    old_coords[:,:3] = moving_mesh.coordinates()

    print "Time"
    for i,t in enumerate(times):
        print "{}/{}".format(t, times[-1])

        moving_mesh.coordinates()[:] = old_coords[:,:3]
        
        u.vector()[:] = data["displacements"][t]
        
        u_diff.vector()[:] = u.vector() - u_prev.vector()
        d = dolfin.interpolate(u_diff, V)
        dolfin.ALE.move(moving_mesh, d)

        
        old_coords = np.ones((moving_mesh.coordinates().shape[0], 4))
        old_coords[:,:3] = moving_mesh.coordinates()
        
        new_coords = np.linalg.inv(F).dot(old_coords.T).T
        moving_mesh.coordinates()[:] = new_coords[:,:3]

        for f in functions.keys():

            if f == "gamma":
        
                if params["gamma_space"] == "regional":
                    rg.vector()[:] = data["gammas"][t]
                    g = dolfin.project(rg.get_function(), gamma_space)
                    functions[f].vector()[:] = g.vector()
                else:
                    functions[f].vector()[:] = data["gammas"][t]
            else:
                
                functions_[f].vector()[:] = features[f][t]
                f_ = dolfin.interpolate(functions_[f], functions[f].function_space())
                functions[f].vector()[:] = f_.vector()

     
        vtk_utils.add_stuff(moving_mesh, vtu_path.format(i), sm,mat,
                            *functions.values())
        
        u_prev.assign(u)
        

    pvd_path = "/".join([outdir, "simulation.pvd"])
    print "Simulation saved at {}".format(pvd_path)
    vtk_utils.write_pvd(pvd_path, fname, time_stamps[:i+1])
   
def make_refined_simulation(params, features, outdir, patient, data):

   

    if not features: return

    import vtk_utils
    

    # Mesh
    mesh_coarse = patient.mesh

    print "before refinement"
    # mesh =  dolfin.adapt(dolfin.adapt(mesh_coarse))
    mesh =  dolfin.adapt(mesh_coarse)
    print "after refinement"
    # Mesh that we move
    moving_mesh = dolfin.Mesh(mesh)


    # The time stamps
    if isinstance(data["gammas"], dict):
        times = sorted(data["gammas"].keys(), key=asint)
    else:
        times = range(len(data["gammas"]))

    if not hasattr(patient, "time"):
        patient.time = range(patient.num_points)
        
    time_stamps = np.roll(patient.time, -np.argmin(patient.time))
    from scipy.interpolate import InterpolatedUnivariateSpline
    s = InterpolatedUnivariateSpline(range(len(time_stamps)), time_stamps, k = 1)
    time_stamps = s(np.array(times, dtype=float))
    
    # Create function spaces
    print "get coarse spaces"
    coarse_spaces = get_feature_spaces(mesh_coarse, params["gamma_space"])
    print "get fine spaces"
    spaces = get_feature_spaces(mesh, params["gamma_space"])
    print "get moving spaces"
    moving_spaces = get_feature_spaces(moving_mesh, params["gamma_space"])
    print "done"

    # Markers
    sm_coarse = dolfin.Function(coarse_spaces["marker_space"])
    sm = dolfin.Function(moving_spaces["marker_space"],
                                name = "AHA-zones")
    sm_coarse.vector()[:] = patient.sfun.array()
    sm_ = dolfin.interpolate(sm_coarse, moving_spaces["marker_space"])
    sm.vector()[:] =sm_.vector()


    functions = {}
    functions_ = {}
    functions_coarse = {}
    for f in features.keys():

        if f in ["displacement", "gamma"]:
            pass

        elif f == "hydrostatic_pressure":
            functions[f] = dolfin.Function(moving_spaces["pressure_space"], 
                                           name=f)
            functions_[f] = dolfin.Function(spaces["pressure_space"], 
                                            name=f)

            functions_coarse[f] = dolfin.Function(coarse_spaces["pressure_space"], 
                                                  name=f)
            
        else:
            functions[f] = dolfin.Function(moving_spaces["cg1"], 
                                          name=f)
            functions_[f] = dolfin.Function(spaces["cg1"], 
                                          name=f)
            functions_coarse[f] = dolfin.Function(coarse_spaces["cg2"], 
                                          name=f)


    # Setup moving mesh
    u_coarse = dolfin.Function(coarse_spaces["displacement_space"])
    u = dolfin.Function(spaces["displacement_space"])
    u_prev = dolfin.Function(spaces["displacement_space"])
    u_diff = dolfin.Function(spaces["displacement_space"])
    # Space for interpolation
    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    # fiber = dolfin.Function(moving_spaces["quad_space"])
   
   
    fname = "refined_simulation_{}.vtu"
    vtu_path = "/".join([outdir, fname])

    print "Time"
    for i,t in enumerate(times):
        print "{}/{}".format(t, times[-1])
        
        u_coarse.vector()[:] = data["displacements"][t]
        print "before interpolation"
        u_ = dolfin.interpolate(u_coarse, spaces["displacement_space"])
        print "after interpolation"
        u.vector()[:] = u_.vector()
        
        u_diff.vector()[:] = u.vector() - u_prev.vector()
        d = dolfin.interpolate(u_diff, V)
        print "before moving mesh"
        dolfin.ALE.move(moving_mesh, d)
        print "after moving mesh"
      

        print "interpolate:"
        for f in functions.keys():
            print f
            functions_coarse[f].vector()[:] = features[f][t]
            f_ = dolfin.interpolate(functions_coarse[f], functions_[f].function_space())
            functions[f].vector()[:] = f_.vector()


     
        vtk_utils.add_stuff(moving_mesh, vtu_path.format(i), sm,
                            *functions.values())
        
        u_prev.assign(u)
        

    pvd_path = "/".join([outdir, "refined_simulation.pvd"])
    print "Simulation saved at {}".format(pvd_path)
    vtk_utils.write_pvd(pvd_path, fname, time_stamps[:i+1])


def save_displacements(params, features, outdir):

    from ..patient_data import FullPatient
    import vtk_utils
    
    patient = FullPatient(**params["Patient_parameters"])

    # Mesh
    mesh = patient.mesh

    spaces = get_feature_spaces(mesh, params["gamma_space"])
    u = dolfin.Function(spaces["displacement_space"])

    path = "/".join([outdir, "displacement.xdmf"])
    f = dolfin.XDMFFile(dolfin.mpi_comm_world(), path)
    times = sorted(features["displacement"].keys(), key=asint)

    for i,t in enumerate(times):

        u.vector()[:] = features["displacement"][t]

        f.write(u, float(t))
        
    
def mmhg2kpa(mmhg):
    """Convert pressure from mmgh to kpa
    """
    return mmhg*101.325/760

def kpa2mmhg(kpa):
    """Convert pressure from kpa to mmhg
    """
    return kpa*760/101.325
def compute_emax(patient, params, val, valve_times):
    
    echo_valve_times  = valve_times#["echo_valve_time"]
              
    pfb = patient.passive_filling_begins
    n = patient.num_points
    es_idx = (echo_valve_times["avc"] - pfb) % n

    matparams = {k:v[0] for k,v in val["material_parameters"].iteritems()}
    
    if val["states"].has_key(str(es_idx)):
        p_es = patient.pressure[es_idx]
        w_es = val["states"][str(es_idx)]
        g_es = val["gammas"][str(es_idx)]
        
        print "es_idx = ", es_idx
        return compute_elastance(w_es, p_es, g_es,
                                 patient,
                                 params,
                                 matparams)
    else:
        return None



def copmute_mechanical_features(patient, params, val, path, keys = None):
    """Compute mechanical features such as stress, strain, 
    works etc, save the output in dolfin vectors to a file, and 
    return a dictionary with average scalar values.

    :param patient: patient data
    :param params: pulse_adjoin.adjoint_contraction_parameters
    :param val: the data
    :param path: path to where to save the output

    """
    
    outdir = os.path.dirname(path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print path

    spaces = get_feature_spaces(patient.mesh, params["gamma_space"])

    dx = dolfin.Measure("dx",subdomain_data = patient.sfun,
                        domain = patient.mesh)
    regions = [int(r) for r in set(patient.sfun.array())]

    meshvols = {"global": float(dolfin.assemble(dolfin.Constant(1.0)*dx))}
    for i in regions:
        meshvols[i] = float(dolfin.assemble(dolfin.Constant(1.0)*dx(i)))

    
    pressures = patient.pressure
    rv_pressures = None if not hasattr(patient, "RVP") else patient.rv_pressure

    W = dolfin.TensorFunctionSpace(patient.mesh, "DG", 0)
    
    ed_point = str(patient.passive_filling_duration) if params["unload"]\
               else str(patient.passive_filling_duration-1)

    # Material parameter
    matparams = {}
    for k,v in val["material_parameters"].iteritems():
        if np.isscalar(v):
            matparams[k] = v
        else:
            if len(v) == 1:
                matparams[k] = v[0]
            else:
                if params["matparams_space"] == "regional":
                    sfun = merge_control(patient, params["merge_passive_control"])
                    par = RegionalParameter(sfun)
                    par.vector()[:] = v
                    matparams[k] = par.get_function()

                else:
                    family, degree =  params["matparams_space"].split("_")
                    V = dolfin.FunctionSpace(patient.mesh, family, int(degree))
                    par = dolfin.Function(V)
              
                    par.vector()[:] = v
                    matparams[k] = par
                
    
    states = val["states"]
    gammas = val["gammas"]
    times = sorted(states.keys(), key=asint)
 
    if not keys:
        keys = default_mechanical_features()


    features = {k.rstrip(":") : [] for k in keys}
    scalar_dict = {str(r):[] for r in ["global"]+regions }
    from copy import deepcopy
    features_scalar = {k.rstrip(":"):deepcopy(scalar_dict) for k in keys}
    
    print("\nExtracting the following features:")
    print("\n".join(keys))

    
    if hasattr(patient, "longitudinal"):
        e_long = patient.longitudinal
        has_longitudinal = True
    else:
        has_longitudinal = False

    if hasattr(patient, "circumferential"):
        e_circ = patient.circumferential

        has_circumferential = True
    else:
        has_circumferential = False

    if hasattr(patient, "radial"):
        e_rad = patient.radial
        has_radial = True
    else:
        has_radial = False

        
    e_f = get_fiber_field(patient)
    
    def get(feature, fun, space, project = True):

        assert space in spaces.keys(), "Invalid space: {}".format(space)
        assert feature in features.keys(), "Invalid feature: {}".format(feature)

        if project:

            f = dolfin.project(fun,spaces[space], solver_type="cg")
            remove_extreme_outliers(f, 300, -300)
        else:
            f = fun

            
        features[feature].append(dolfin.Vector(f.vector()))
        
        if feature != "displacement":


            if feature == "gamma":
                regional = get_regional(dx, f, [f.vector().array()], regions)
                scalar = get_global(dx, f, [f.vector().array()], regions)
            else:
                regional = get_regional_quad(dx, fun, regions)
                scalar = get_global_quad(dx, fun)
  
            for i,r in enumerate(regions):
                features_scalar[feature][str(r)].append(regional[i])
                
            features_scalar[feature]["global"].append(scalar)
        
        return f
    

    for t in times:

        print("\tTimepoint {}/{} ".format(t, len(times)-1))
        state = states[t]
        gamma = gammas[t]
        pressure = pressures[int(t)]
        rv_pressure = None if rv_pressures is None else rv_pressures[int(t)]
        
        solver, p_lv = get_calibrated_solver(state, pressure,
                                             gamma,
                                             patient,
                                             params, 
                                             matparams, rv_pressure)

        u,p = solver.get_state().split(deepcopy=True)


        post = solver.postprocess()

        w_ed = dolfin.Function(solver.get_state().function_space())
        w_ed.vector()[:] = states[ed_point]
        u_ed, _ = w_ed.split(deepcopy=True)
        F_ed = dolfin.Identity(3) + dolfin.grad(u_ed)
        F = dolfin.Identity(3) + dolfin.grad(u)
        
        for k in keys:

            k1, k2 = k.split(":")

            if k1 == "displacement":
                get("displacement", u, "displacement_space", False)
                
            elif k1 == "gamma":
                gamma = solver.get_gamma()
                gamma.vector()[:] = np.multiply(params["T_ref"], gamma.vector().array())

                get("gamma", gamma, "gamma_space", False)

            elif k1 == "hydrostatic_pressure":
                get("hydrostatic_pressure", p, "pressure_space", False)

            elif k1 == "I1":

                I1 = solver.parameters["material"].active._I1(F)
                get("I1", I1, "cg2")
                
            elif k1 == "I1e":

                I1e = solver.parameters["material"].active.I1(F)
                get("I1e", I1e, "cg2")

            elif k1 == "I4f":

                f0 = solver.parameters["material"].get_component("fiber")
                I4f = solver.parameters["material"].active._I4(F, f0)
                get("I4f", I4f, "cg2")
                
            elif k1 == "I4fe":

                I4fe = solver.parameters["material"].active.I4(F, "fiber")
                get("I4fe", I4fe, "cg2")
                
            else:

                if k2 == "longitudinal":
                    if not has_longitudinal: continue
                    e = e_long
                elif k2 == "radial":
                    if not has_radial: continue
                    e = e_rad
                elif k2 == "circumferential":
                    if not has_circumferential: continue
                    e = e_circ
                else:
                    e = e_f

                if k1 == "green_strain":

                    E = dolfin.project(post.GreenLagrange(F_ref=F_ed), W)
                    Ef = dolfin.inner(E*e, e)
                    
                    get(k, Ef, "cg2")

                    
                    
                elif k1 == "cauchy_stress":

                    Tf = solver.postprocess().cauchy_stress_component(e, deviatoric=False)
                    get(k, Tf, "cg2")
                    
                    
                elif k1 == "cauchy_dev_stress":

                    Tf = solver.postprocess().cauchy_stress_component(e, deviatoric=True)
                    get(k, Tf, "cg2")

                elif k1 == "almansi_strain":
                    Ef = solver.postprocess().almansi_strain_component(e, F_ref=F_ed)
                    get(k, Ef, "cg2")
                    


    from load import save_dict_to_h5
    save_dict_to_h5(features, path)

    return features_scalar


def get_solver(matparams, patient, gamma, params):

    from ..setup_optimization import make_solver_parameters
    from ..lvsolver import LVSolver

    solver_parameters, pressure, paramvec= make_solver_parameters(params, patient,
                                                                  matparams, gamma)
    return LVSolver(solver_parameters), pressure


def get_calibrated_solver(state_arr, pressure, gamma_arr,
                          patient, params, matparams, rv_pressure = None):
    
    if params["gamma_space"] == "regional":
        sfun = merge_control(patient, params["merge_active_control"])
        gamma = RegionalParameter(sfun)
        gamma_tmp = RegionalParameter(sfun)
    else:
        gamma_space = dolfin.FunctionSpace(patient.mesh, "CG", 1)
        gamma_tmp = dolfin.Function(gamma_space, name = "Contraction Parameter (tmp)")
        gamma = dolfin.Function(gamma_space, name = "Contraction Parameter")

    solver, p_expr = get_solver(matparams, patient, gamma, params)

    
    gamma_tmp.vector()[:] = gamma_arr
    gamma.assign(gamma_tmp)


    w = dolfin.Function(solver.get_state_space())
    w.vector()[:] = state_arr

    solver.reinit(w)

    
    return solver, p_expr

def remove_extreme_outliers(fun, ub=None, lb=None):
    """
    Set all values that are larger than ub to ub, 
    and set all values that are lower than lb to lb.

    fun : dolfin.Function
        The function from which you want to remove extreme outliers
    ub : float
        Upper bound
    lb : float
        Lower bound
    
    """
    if lb is None: lb = -np.inf
    if ub is None: ub = np.inf
    fun.vector()[fun.vector().array() > ub] = ub
    fun.vector()[fun.vector().array() < lb] = lb

    

def smooth_from_points(V, f0, nsamples = 10, method="interpolate") :
    """
    Smooth f0 by interpolating f0 into V by using radial basis functions
    for interpolating scattered data using nsamples.
    The higher values of nsamples, the more you smooth.
    
    This is very useful is e.g f0 is a function in a
    quadrature space

    Parameters
    ----------

    V : dolfin.FunctionSpace
        The space for the function to be returned
    f0 : dolfin.Function
        The function you want to smooth
    nsamples : int (optional)
        For each degree of freedom in V, use nsamples to
        build the radial basis function. Default = 20.
    method : str (optional)
        Method for smoothing. Either `interpolate` using
        radial basis functions, or `average`, or `median`

    Returns
    -------
    f : dolfin.Function
        The function f0 interpolated into V
        using radial basis funcitions for interpolating 
        scattered data

    """

    from scipy.spatial import cKDTree
    from scipy.interpolate import Rbf
    import numpy as np
    # points for f0
    V0 = f0.function_space()
    # xyz = V0.dofmap().tabulate_all_coordinates(V0.mesh()).reshape(-1, 3)
    xyz =  V0.tabulate_dof_coordinates().reshape((-1, 3))
    f0val = f0.vector().array()
    tree = cKDTree(xyz)

    # coordinate of the dofs
    # coords = V.dofmap().tabulate_all_coordinates(V.mesh()).reshape(-1, 3)
    coords =  V.tabulate_dof_coordinates().reshape((-1, 3))
    f = dolfin.Function(V)

    
    for idx in xrange(0, V.dim()) :
        v = coords[idx,:]
        samples_rad, samples_idx = tree.query(v, nsamples)
        a =  xyz[samples_idx,:]
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        _, inds = np.unique(b, return_index=True)
        c = a[inds]
        s_idx=samples_idx[inds]

        
        xx, yy, zz = np.split(c, 3, axis=1)
        fvals = f0val[s_idx]

        if method == "interpolate":
            rbf = Rbf(xx, yy, zz, fvals, function= 'gaussian')# function='linear')
            val = float(rbf(v[0], v[1], v[2]))

        elif method == "average":
            # Remove outliers (include only the values within 1 std)
            fvals_ = fvals[abs(fvals - np.mean(fvals)) < 2 * np.std(fvals)]
            if len(fvals_) > 0:
                val = np.mean(fvals_)
            else:
                val = np.median(fvals)
                
        elif method == "median":
            val = np.median(fvals)

            
        f.vector()[idx] = val
        
        
    return f



def localproject(fun, V) :
    """
    Cheaper way of projecting than regular projections.
    This is useful if you have many degrees of freedom.
    For more info, see dolfin.LocalSolver.

    Parameters
    ---------- 
    fun : dolfin.Function
        The function you want to project
    V : dolfin.FunctionSpace
        The you want to project into
    
    Returns
    -------
    res : dolfin.Function
        fun projected into V
    
    """
    a = dolfin.inner(dolfin.TestFunction(V), dolfin.TrialFunction(V)) * dolfin.dx
    L = dolfin.inner(dolfin.TestFunction(V), fun) * dolfin.dx
    res = dolfin.Function(V)
    solver = dolfin.LocalSolver(a,L)
    solver.solve_local_rhs(res)
    return res


def setup_bullseye_sim(bullseye_mesh, fun_arr):
    V = FunctionSpace(bullseye_mesh, "DG", 0)
    dm = V.dofmap()
    sfun = MeshFunction("size_t", bullseye_mesh, 2, bullseye_mesh.domains())

    funcs = []
    for time in range(len(fun_arr)):

        fun_tmp = Function(V)
        arr = fun_arr[time]

        for region in range(17):

                vertices = []

                for cell in cells(bullseye_mesh):

                    if sfun.array()[cell.index()] == region+1:

                        verts = dm.cell_dofs(cell.index())

                        for v in verts:
                            # Find the correct vertex index 
                            if v not in vertices:
                                vertices.append(v)

                fun_tmp.vector()[vertices] = arr[region]
        funcs.append(Vector(fun_tmp.vector()))
    return funcs
    
