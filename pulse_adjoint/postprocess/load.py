#!/usr/bin/env python
"""
This script includes functionality to load the results obtained using
pulse adjoint into a a format that are easier to handle.

There are still some results that are not loaded yet.
This included timings and convergence details. 
This will be included later. 
"""
#!/usr/bin/env python
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
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
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
import os, h5py
from copy import deepcopy
import numpy as np
from .args import *
import utils


attributes = ["pressure", "volume", "RVP", "RVV",
              "passive_filling_duration",
             "passive_filling_begins",
             "num_points", "num_contract_points",
             "time"]
dicts = ["work", "strain", "strain_3d", "original_strain"]
strain_dirs = ["circumferential", "radial", "longitudinal"]
patient_paths = ["echo_path", "pressure_path", "mesh_path"]
def lookup(dic, key, *keys):
    if keys:
        return lookup(dic.get(key, {}), *keys)
    return dic.get(key)
def load_measured_strain(d, patient, key = "measured_strain"):

    d[key] = {}
    d[key] = {k:{} for k in strain_dirs}
    for k in d[key].keys():
        d[key][k] ={r:{} for r in patient.strain.keys()}
    
    for k1, v1 in patient.strain.iteritems():
        
        if isinstance(v1, dict):
            for k2, v2 in v1.iteritems():
                if isinstance(v2, np.ndarray):
                    d[key][k2][k1] = v2.tolist()
                else:
                    d[key][k2][k1] = v2

        else:
            assert isinstance(v1, (list, np.ndarray))
            for i, k2 in enumerate(strain_dirs):
                d[key][k2][k1] = np.transpose(v1)[i].tolist()

            
def get_relative_paths(params):

    for path_key in ["echo_path", "mesh_path", "pressure_path"]:
        path = params[path_key]
        if path == "": continue
        path_split = path.split("/")
        path_found = False
        for j in range(len(path_split)):
            if os.path.isfile("/".join(path_split[-j:])):
                params[path_key] = "/".join(path_split[-j:])
                path_found = True

        if not path_found:
            for j in range(len(path_split)):
                if os.path.isfile("/".join(["../"]+path_split[-j:])):
                    params[path_key] = "/".join(["../"]+path_split[-j:])
                    path_found = True
            
            if not path_found:
                msg = ("Path {} not found. ".format(path_key) +
                       "Pleas specify this path in the parameter file."
                       "\nDefault path is {}".format(path))
                raise IOError(msg)
        
    return params



def save_parameters(params, fname, key):

    from dolfin import Parameters
    import yaml
    if isinstance(params, Parameters):
        params = params.to_dict()

    msg = "Illegal type for parameters {}".format(type(params))
    assert isinstance(params, dict), msg
    
    if os.path.isfile(fname):
        with open(fname, "r") as f:
            d = yaml.load(f)
    else:
        d = {}

    d[key] = params

    with open(fname, "wb") as f:
        yaml.dump(d, f, default_flow_style=False)

def load_parameters(fname, key=None):
    
    import yaml
    if not os.path.isfile(fname):
        raise IOError("The file {} does not exist".format(fname))

    with open(fname, "r") as f:
        d = yaml.load(f)

    if key is None: return d
    
    if d.has_key(key):
        return d[key]
    else:
        msg = "Parameters does not have key {}. Possible keys are {}".format(key, d.keys())
        raise KeyError(msg)
        

def load_patient_data(h5name, h5group):
    from mesh_generation import load_geometry_from_h5
    from ..patient_data import FullPatient

    patient = FullPatient(init=False)

    geo = load_geometry_from_h5(h5name, h5group)

    for k,v in geo.__dict__.iteritems():
        if k not in patient_paths:
            setattr(patient, k, v)

    data = load_dict_from_h5(h5name, "/".join([h5group, "data"]))

    for attr in attributes:
        if data.has_key(attr):
            
            if len(data[attr]) == 1:
                setattr(patient, attr, int(data[attr][0]))
            else:
                setattr(patient, attr, data[attr])

    for d in dicts:
        if data.has_key(d):
            q = {}
            for k,v in data[d].iteritems():
                q[int(k)] = np.array(h5dict_to_list(v))
                
            setattr(patient, d, q)

    paths = {}
    for p in patient_paths:
        if hasattr(geo, p):
            paths[p] = getattr(geo, p)
            
    setattr(patient, "paths", paths)


    return patient

def save_patient_to_h5(patient, h5name, h5group):
    

    if hasattr(patient, 'mesh'):
        mesh = getattr(patient, 'mesh')
    else:
        raise ValueError("Patient has no mesh!")

    if hasattr(patient, 'markers'):
        markers = getattr(patient, 'markers')
    else:
        raise ValueError("Patient has no markers")

    fields_names = ['fiber', 'sheet', 'sheet_normal']
    local_basis_names = ['circumferential','radial', 'longitudinal']
    
    fields = []
    local_basis = []

    for f in fields_names:
        if hasattr(patient, f):
            fields.append(getattr(patient, f))

    for l in local_basis_names:
        if hasattr(patient, l):
            local_basis.append(getattr(patient, l))

    h5py_functions={}
    
    for attr in attributes:
        if hasattr(patient, attr):
            f = getattr(patient, attr)
            
            if np.isscalar(f):
                h5py_functions[attr] = np.array([f])
            elif isinstance(f, (list, np.ndarray)):
                h5py_functions[attr] = np.array(f)
            else:
                raise ValueError("Unknown type {}".format(type(f)))
        else:
            logger.info("Patient do not have attribute {}".format(attr))

    for attr in dicts:
        if hasattr(patient, attr):
            f = getattr(patient, attr)
            h5py_functions[attr] = f

    if hasattr(patient, 'valve_times'):
        h5py_functions['valve_times'] = getattr(patient, 'valve_times')

    other_functions={}
    if hasattr(patient, "original_geometry"):
        other_functions["original_geometry"] = getattr(patient, "original_geometry")


    if hasattr(patient, "paths"):
        other_attributes = getattr(patient, "paths")
    else:
        other_attributes = {}

    logger.info("Save patient data to {}:patient".format(h5name))
    from mesh_generation import save_geometry_to_h5
    save_geometry_to_h5(mesh, h5name, h5group = h5group,
                        markers = markers,
                        fields = fields,
                        local_basis = local_basis,
                        comm = dolfin.mpi_comm_world(),
                        other_functions = other_functions,
                        other_attributes = other_attributes)

    save_dict_to_h5(h5py_functions, h5name, "/".join([h5group, "data"]), False, True)

def get_value_from_h5dict(d):

    if isinstance(d, np.ndarray):
        return d
    elif isinstance(d, dict):
        if len(d.keys()) == 0:
            logger.warning("Dictionary is empty")
            return []
        
        elif len(d.keys()) == 1:
            return d[d.keys()[0]]

        else:
            if "vector_0" in d.keys():
                return d["vector_0"]

            else:
                msg = ("Do not know what to return"+
                       "Possible keys are {}".format(d.keys()))
                raise ValueError(msg)
            
    else:
        msg = "Unknown type {}".format(type(d))
        raise ValueError(msg)

def h5dict_to_list(d):

    keys = sorted(d.keys(), key = lambda t: int(t))
    lst = []
    for k in keys:
        lst.append(get_value_from_h5dict(d[k]))

    return lst

def flatten_dict(d, toint = True):

    def flatten(di):

        if isinstance(di, np.ndarray):
            if toint:
                dii = int(di[0])
            else:
                dii = di[0]
                
        elif isinstance(di, dict):
            dii = {}
            for k, v in di.iteritems():
                dii[k] = flatten(v)
            
        else:
            raise ValueError("Unkown type {}".format(type(di)))

        return dii
    
    return flatten(d)

def get_patient_geometry_from_results(params, has_unloaded = False):

    from ..setup_optimization import initialize_patient_data, update_unloaded_patient
    patient_params = get_relative_paths(params["Patient_parameters"])
    patient = initialize_patient_data(patient_params)

    if params["unload"] and has_unloaded:
        patient = update_unloaded_patient(params, patient)

    return patient

def get_unloaded_data(params):

    if not os.path.isfile(params["sim_file"]):
        raise IOError("File {} does not exist".format(params["sim_file"]))


    all_data = load_dict_from_h5(params["sim_file"])

    keys = sorted([a for a in all_data.keys() if a.isdigit()],
                  key = lambda t: int(t))

    simulated_volumes = []
    optimal_controls = []
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    for k in keys[:-1]:

        target = np.array([a[0] for a in h5dict_to_list(all_data[k]["passive_inflation"]["volume"]["target"])])
        simulated = np.array([a[0] for a in h5dict_to_list(all_data[k]["passive_inflation"]["volume"]["simulated"])])
        simulated_volumes.append(simulated)


        initial_control = all_data[k]["passive_inflation"]["initial_control"]["0"]
        optimal_control = all_data[k]["passive_inflation"]["optimal_control"]["vector_0"]

        optimal_controls.append(optimal_control[0])
        
        print target
        print simulated
        print initial_control
        print optimal_control
        # if k == "1":


        if int(k) > 0:
            v0, v1 = np.array(simulated_volumes).T[-1][-2:]
            v_target = target[-1]
        
            delta = (v_target - v0)/(v1-v0)
            a0, a1 = optimal_controls[-2:]
            a = (1-delta)*a0 + delta*a1
            print "delta = ", delta
            print "a = ", a

            
        print "#"*40
        ax.plot(float(k)*np.ones(len(simulated)), simulated, "o")

    ax.plot((float(k)+1)*np.ones(len(simulated)), target, "x")
    plt.show()
    
    

    

def load_dict_from_h5(fname, h5group = ""):
    """
    Load the given h5file into
    a dictionary
    """
    import h5py
    assert os.path.isfile(fname), \
        "File {} does not exist".format(fname)

    # Just some error handling in case file is broken
    try: f = h5py.File(fname, "r")
    except: return {}
    f.close()
    
    with h5py.File(fname, "r") as h5file:

        def h52dict(hdf):
            if isinstance(hdf, h5py._hl.group.Group):
                t = {}
        
                for key in hdf.keys():
                    t[str(key)] = h52dict(hdf[key])
    
                
            elif isinstance(hdf, h5py._hl.dataset.Dataset):
                t = np.array(hdf)

            return t

        if h5group != "":
            if h5group in h5file:
                d = h52dict(h5file[h5group])
            else:
                msg = "h5group {} does not exist in h5file {}".format(fname, h5group)
                logger.warning(msg)
                return None
        else:
            d = h52dict(h5file)

    return d

def load_geometry_and_microstructure_from_results(params):

    from mesh_generation.mesh_utils import load_geometry_from_h5
    if params["unload"]:
        original = load_geometry_from_h5(params["sim_file"])
        unloaded = load_geometry_from_h5(params["sim_file"], "unloaded")
        unloaded.original_mesh = original.mesh
        return unloaded
    else:
        return load_geometry_from_h5(params["sim_file"])
    

def save_dict_to_h5(d, h5name, h5group = "",
                    overwrite_file = True, overwrite_group=True):
    """Create a HDF5 file and put the
    data in the dictionary in the 
    same hiearcy in the HDF5 file
    
    Assume leaf of dictionary is either
    float, numpy.ndrray, list or 
    dolfin.GenericVector.

    :param d: Dictionary to be saved
    :param h5fname: Name of the file where you want to save
    

    .. note:: 

        Works only in serial
    
    """
    import h5py
    if overwrite_file:
        if os.path.isfile(h5name):
            os.remove(h5name)

    file_mode = "a" if os.path.isfile(h5name) and not overwrite_file else "w"
   
    # IF we should append the file but overwrite the group we need to
    # check that the group does not exist. If so we need to open it in
    # h5py and delete it.
    if file_mode == "a" and overwrite_group:
        with h5py.File(h5name) as h5file:
            if h5group in h5file:
                logger.debug("Deleting existing group: '{}'".format(h5group))
                del h5file[h5group]
                    

    with dolfin.HDF5File(dolfin.mpi_comm_world(), h5name, file_mode) as h5file:

        def dict2h5(a, group):

            for key, val in a.iteritems():
                
                subgroup = "/".join([group, str(key)])

                if isinstance(val, dict):
                    dict2h5(val, subgroup)
                
                elif isinstance(val, (list,  np.ndarray, tuple)):

                    if len(val) == 0:
                        # If the list is empty we do nothing
                        pass
                    
                    elif isinstance(val[0], (dolfin.Vector, dolfin.GenericVector)):
                        for i, f in enumerate(val):
                            h5file.write(f, subgroup + "/{}".format(i))

                    elif isinstance(val[0], (float, int)):
                        h5file.write(np.array(val, dtype=float), subgroup)

                    elif isinstance(val[0], (list, np.ndarray, tuple, dict)):
                        # Make this list of lists into a dictionary
                        f = {str(i):v for i,v in enumerate(val)}
                        dict2h5(f, subgroup)                
                        
                    else:
                        raise ValueError("Unknown type {}".format(type(val[0])))
                    
                elif isinstance(val, (float, int)):
                    h5file.write(np.array([float(val)]), subgroup)

                elif isinstance(val, (dolfin.Vector, dolfin.GenericVector)):
                    h5file.write(val, subgroup)

                else:
                    raise ValueError("Unknown type {}".format(type(val)))

        dict2h5(d, h5group)





def get_data(params, patient=None):
    """Get the data from the pulse adjoint results.
    Load the states, displacements, material parameters
    and the activation parameter gamma

    It also removes points where interpolation is performed
    in order to obtain convergence, so that the simulated data
    agrees with the measured data. 

    .. note

       More data such as timings and convergece will be added later

    :param params: adjoint contraction parameters
    :param patient: patient class
    :returns: the data
    :rtype: dict

    """
    

    
    if not os.path.isfile(params["sim_file"]):
        raise IOError("File {} does not exist".format(params["sim_file"]))


    
    ####### Containers and keys
    all_data = load_dict_from_h5(params["sim_file"])

    
    passive = {} if not all_data.has_key("passive_inflation") else all_data["passive_inflation"]
    active = {} if not all_data.has_key("active_contraction") else all_data["active_contraction"]
    active_keys = sorted(active.keys(),
                         key = lambda t : int(t.rsplit("contract_point_")[-1]))


    main_active_group = "active_contraction"
    passive_group = "passive_inflation"
    active_group = "/".join([main_active_group, "contract_point_{}"])
    

    opt_res = {"run_time":[],
               "nit":[],
               "nfev":[],
               "njev":[],
               "ncrash":[],
               "func_vals": [],
               "forward_times":[],
               "backward_times":[]}
    
    data = {"states":[],
            "gammas":[],
            "displacements":[],
            "passive_optimization_results":deepcopy(opt_res),
            "active_optimization_results":{}}

    

    rv = params['Optimization_targets']['rv_volume']

    ###### Create proper functions and objects
    
    if patient is None:
        patient = get_patient_geometry_from_results(params, all_data.has_key("unloaded"))

    from utils import init_spaces
    spaces = init_spaces(patient.mesh, params["gamma_space"])

    from pulse_adjoint.setup_optimization import RegionalParameter, merge_control
    if params["gamma_space"] == "regional":
        sfun = merge_control(patient, params["merge_active_control"])
        gamma = RegionalParameter(sfun)
    else:
        gamma = dolfin.Function(spaces["gamma_space"], name = "Contraction Parameter")
        
    state = dolfin.Function(spaces["state_space"], name = "State")
    u= dolfin.Function(spaces["displacement_space"], name = "u")

    # Done loading data using h5py. Now load dolfin data
    # opt_keys = ["nfev", "njev", "nit", "run_time", "ncrash"]
    matlst = params["Fixed_parameters"].keys()
    npassive = sum([not v for v in params["Fixed_parameters"].values()])
    
   
    if npassive == 1:

        if params["matparams_space"] == "regional":
            sfun = merge_control(patient, params["merge_passive_control"])
            paramvec = RegionalParameter(sfun)
        else:
            family, degree = params["matparams_space"].split("_")                
            paramvec = dolfin.Function(dolfin.FunctionSpace(patient.mesh, family, int(degree)), name = "matparam vector")
    else:
        paramvec = dolfin.Function(dolfin.VectorFunctionSpace(patient.mesh, "R", 0, dim = npassive), name = "matparam vector")


    if isinstance(params["Material_parameters"], dolfin.Parameters):
        matparams = params["Material_parameters"].to_dict()
    else:
        matparams = params["Material_parameters"]
        
    def read_matparam(h5file, group, paramvec, all_data=None):
        paramvec = dolfin.Function(paramvec.function_space())
        try:
            h5file.read(paramvec,  group)

        except:
            glst = group.split("/")
            pararr = lookup(all_data, *glst)["0"]

        else:
            if len(paramvec.vector()) == npassive:
                pararr = paramvec.vector().array()
            else:
                if npassive == 1:
                    pararr = [paramvec.vector().array()]
                else:
                    pararr = [par.vector().array() for par in paramvec.split(deepcopy=True)]
        finally:
            fixed_idx = np.nonzero([not params["Fixed_parameters"][k] for k in matlst])[0]

            for it, idx in enumerate(fixed_idx):
                par = matlst[idx]
                matparams[par] = pararr[it]
                
            from copy import deepcopy
            return deepcopy(matparams)


        
    # Make sure to load results from uloading phase first
    if params["unload"]:


        data["unload"] = {"material_parameters":{},
                          "reference_volume":{},
                          "ed_volume":{},
                          # "optimal_material_parameters":{},
                          "backward_displacement":{},
                          "unloaded_volumes":{},
                          "func_vals":{},
                          "ed_volumes":{},
                          "optimized_volumes":{}}
        if rv:
            data["unload"]["optimized_rv_volumes"] = {}
            data["unload"]["reference_rv_volume"] = {}
            data["unload"]["ed_rv_volume"] = {}

        unload_iters = []
        for k in all_data.keys():
            if k.isdigit():
                if all_data[k].has_key("passive_inflation"):
                    unload_iters.append(k)
                else:
                    msg = ("\nWARNING:\nPassive inflation for iteration {} ".format(k)+
                           "does not exist.\nSimulation most likely failed here\n")
                    logger.info(msg)

     
        unload_iters = sorted(unload_iters, key = lambda t : int(t))

        if not unload_iters or not all_data[unload_iters[0]].has_key("passive_inflation"):
            return {}, patient
        
        data["unload"]["target_volumes"] = all_data[unload_iters[0]]["passive_inflation"]["volume"]["target"]
        if rv:
            data["unload"]["target_rv_volumes"] = all_data[unload_iters[0]]["passive_inflation"]["rv_volume"]["target"]
        
        unload_subiters = {}
        for k in unload_iters:
            its = []
            for i in all_data[k].keys():
                if i.isdigit(): its.append(i)
            unload_subiters[k] = sorted(its, key = lambda t : int(t))

            try:
                data["unload"]["func_vals"][k] = all_data[k]["passive_inflation"]["optimization_results"]["func_vals"]
            except:
                pass
            data["unload"]["optimized_volumes"][k] = all_data[k]["passive_inflation"]["volume"]["simulated"]
            if rv:
                data["unload"]["optimized_rv_volumes"][k] = all_data[k]["passive_inflation"]["rv_volume"]["simulated"]


        with dolfin.HDF5File(dolfin.mpi_comm_world(), params["sim_file"], "r") as h5file:

            V_back = dolfin.VectorFunctionSpace(patient.mesh, "CG", 1)
            u_back = dolfin.Function(V_back)
            refmesh = dolfin.Mesh()
            
            def get_backward_displacement(k):
                h5file.read(u_back, "/".join([k, "unloaded", "backward_displacement"]))
                return dolfin.Vector(u_back.vector())

            def get_reference_volume(k, marker):
                h5file.read(refmesh,  "/".join([k, "unloaded", "geometry", "mesh"]), True)
                ffun = dolfin.MeshFunction("size_t", refmesh, 2, refmesh.domains())
                return utils.compute_inner_cavity_volume(refmesh, ffun, marker)

            def get_ed_volume(k, marker):
                k1 = unload_subiters[k][-1]
                h5file.read(refmesh,  "/".join([k, k1, "ed_geometry"]), True)
                ffun = dolfin.MeshFunction("size_t", refmesh, 2, refmesh.domains())
                return utils.compute_inner_cavity_volume(refmesh, ffun, marker)

            if "initial_passive" in all_data.keys():

                for a_, l_ in zip(["initial_control", "optimal_control"],[-2,-1]):
                    group = "/".join(["initial_passive", passive_group, a_])
                    mat = read_matparam(h5file, group, paramvec, all_data)
                    data["unload"]["material_parameters"][l_]=mat

            for k in unload_iters:
                group = "/".join([k, passive_group, "optimal_control"])
              
                mat = read_matparam(h5file, group, paramvec)
                print group
                print mat
                try: data["unload"]["material_parameters"][k] = mat
                except: pass

                try:
                    data["unload"]["backward_displacement"][k] = get_backward_displacement(k)
                except: pass

                try:
                    data["unload"]["reference_volume"][k] = get_reference_volume(k, 30)
                except: pass
                
                if rv:
                    try:
                        data["unload"]["reference_rv_volume"][k] = get_reference_volume(k, 20)
                    except: pass

                try:
                    data["unload"]["ed_volume"][k] = get_ed_volume(k, 30)
                except: pass
                if rv:
                    try:
                        data["unload"]["ed_rv_volume"][k] = get_ed_volume(k, 20)
                    except: pass
                

    
    

    if params["unload"]:
        measured_pressure = [0.0] + np.array(patient.pressure).tolist()
    else:
        measured_pressure = np.subtract(patient.pressure,
                                        patient.pressure[0])

    if not passive:
        msg = "No passive data found. Return... "
        print(msg)
        return data, patient
    
    pressures = passive["bcs"]["pressure"]
    for i, k in enumerate(active_keys):
        pressures = np.append(pressures,active[k]["bcs"]["pressure"][1:])

     
    k = 0
    interpolation_points = []
    
    for i, p in enumerate(pressures):
        # print "k = ", k ,"p = ", p, "measured = ", measured_pressure[k]
        if p == measured_pressure[k]:
            k += 1
        else:
            interpolation_points.append(i)

    with h5py.File(params["sim_file"], "r") as h5file:

        if "optimization_results" in h5file[passive_group]:
            for k in opt_res.keys():
            
                if k in h5file[passive_group]["optimization_results"]:
                    data["passive_optimization_results"][k] = h5file[passive_group]["optimization_results"][k][:]

        for p in range(len(active_keys)):
                
            if active_group.format(p) in h5file:
                opt_res_ = deepcopy(opt_res)
                for k in opt_res.keys():
                    if "optimization_results" in h5file[active_group.format(p)]:
                        opt_res_[k] = h5file[active_group.format(p)]["optimization_results"][k][:]

                data["active_optimization_results"]["contract_point_{}".format(p)] = opt_res_


    with dolfin.HDF5File(dolfin.mpi_comm_world(), params["sim_file"], "r") as h5file:

        
        it = 0
        print "Reading passive data"

        # Material parameters
        mat = read_matparam(h5file, "/".join([passive_group, "optimal_control"]), paramvec)
        print mat
        data["material_parameters"] = mat

                
                
        
        N = patient.passive_filling_duration+1 if params["unload"] else patient.passive_filling_duration
               
        for p in range(N):
            
            if not it in interpolation_points:
                h5file.read(state, "/".join([passive_group, "states", str(p)]))
                h5file.read(u, "/".join([passive_group, "displacement", str(p)]))

                data["displacements"].append(dolfin.Vector(u.vector()))
                data["states"].append(dolfin.Vector(state.vector()))
                data["gammas"].append(dolfin.Vector(gamma.vector()))

       
            it += 1

        for p in range(len(active_keys)):
            
            if not it in interpolation_points:
                h5file.read(state, "/".join([active_group.format(p), "states/0"]))
                h5file.read(u, "/".join([active_group.format(p), "displacement/0"]))
                h5file.read(gamma, "/".join([active_group.format(p), "optimal_control"]))

             
                data["displacements"].append(dolfin.Vector(u.vector()))
                data["states"].append(dolfin.Vector(state.vector()))
                data["gammas"].append(dolfin.Vector(gamma.vector()))
            it += 1


    return data, patient


def get_kwargs(patient, params):
    kwargs = init_spaces(patient.mesh, params["gamma_space"])
    kwargs["longitudinal"] = patient.longitudinal
    kwargs["radial"] = patient.radial
    kwargs["circumferential"] = patient.circumferential
    
    
    # Add some extra stuff to arguments
    kwargs["mesh"] = patient.mesh
    kwargs["strain_markers"] = patient.sfun
    kwargs["facets_markers"] = patient.ffun
    
    kwargs["dx"] = Measure("dx", subdomain_data = patient.sfun,
                           domain = patient.mesh)
    
    kwargs["segmentation"] = patient.get_original_echo_surfaces()
    kwargs["segmentation"]["transformation"] = patient.transformation_matrix
    kwargs["ENDO"] = patient.ENDO
    kwargs["EPI"] = patient.EPI
    kwargs["BASE"] = patient.BASE

    return kwargs


def load_measured_strain_and_volume(patient, params, num_points = None):


    from pulse_adjoint.setup_optimization import get_measurements
    params["phase"] = "all"
    
    data = get_measurements(params, patient)
    
    # Some indices
    passive_filling_duration = patient.passive_filling_duration
    if hasattr(patient, "passive_filling_begins"):
        pfb = patient.passive_filling_begins
    else:
        pfb = 0

    
    num_contract_points = patient.num_contract_points
    # Total points 
    num_points = num_contract_points + passive_filling_duration
    
    # Time stamps
    if hasattr(patient, "time"):
        time_stamps = np.subtract(patient.time,patient.time[0])
        time_stamps = time_stamps[:num_points]
    else:
        time_stamps = range(num_points)

        
    data["time_stamps"] = time_stamps
    data["num_points"] = num_points
    data["passive_filling_begins"] = pfb
    data["passive_filling_duration"] = passive_filling_duration
    data["num_contract_points"] = num_contract_points

    for k, v in data.iteritems():
        setattr(patient, k, v)

    
    return patient
