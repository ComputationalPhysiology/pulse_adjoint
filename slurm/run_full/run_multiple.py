#!/usr/bin/env python



import os
from numpy import logspace, multiply
from itertools import product
import yaml

patients = ["JohnDoe"]
filepath= os.path.dirname(os.path.abspath(__file__))
OUTPATH = os.path.join(filepath,"results/patient_{}/active_model_{}/gamma_space_{}")

def main():

    ### Cobinations ###
    unload = True

    # Patients


    # Active model
    active_models = ["active_strain", "active_stress"]

    # Space for contraction parameter
    gamma_spaces = ["regional", "CG_1"]
    #gamma_spaces = ["CG_1"]
    matparams_space = "R_0"

    ### Fixed for all runs ###
    opttargets = {"volume":True,
                  "rv_volume": False,
                  "regional_strain":True,
                  "full_strain":False,
                  "GL_strain":False,
                  "GC_strain":False,
                  "displacement":False}
    
    optweight_active = {"volume":0.95, 
                        "regional_strain": 0.05, 
                        "regularization": 0.1}
    optweight_passive = {"volume":1.0,
                         "regional_strain": 0.0}
                  
    fiber_angle_epi = -60
    fiber_angle_endo = 60
    
    
    # Use gamma from previous iteration as intial guess
    initial_guess ="previous"
    # Optimize material parameters or use initial ones
    optimize_matparams = True

 
    # Spring constant at base
    base_spring_k = 100.0
    pericardium_spring = 0.0
    # Initial material parameters
    material_parameters = {"a":2.28, "a_f":1.685, "b":9.726, "b_f":15.779}


    ### Run combinations ###
 
    # Find all the combinations
    comb = list(product(patients,active_models, gamma_spaces))

    # Directory where we dump the paramaters
    input_directory = "input"
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)

    fname = input_directory + "/file_{}.yml"
    
    # Find which number we 
    t = 1
    while os.path.exists(fname.format(t)):
        t += 1
    t0 = t


    for c in comb:


        params = {"Patient_parameters":{},
                  "Optimization_parameters":{}, 
                  "Optimization_targets":{},
                  "Unloading_parameters":{"unload_options":{}},
                  "Active_optimization_weigths": {},
                  "Passive_optimization_weigths":{},
                  "Optimization_parameters": {}}

        params["active_model"] = c[1]
        params["gamma_space"] = c[2]
        params["optimize_matparams"] = optimize_matparams
        params["matparams_space"] = matparams_space
        params["log_level"] = 20
        params["passive_weights"] = "-1"
        params["initial_guess"] = initial_guess
        params["Patient_parameters"]["patient"] = c[0]
        params["active_relax"] = 1.0
        params["base_spring_k"] = base_spring_k
        params["pericardium_spring"] = pericardium_spring
        params["passive_relax"] = 1.0
        params["unload"] = unload
        
        if unload:
            params["Unloading_parameters"]["maxiter"] = 10
            params["Unloading_parameters"]["tol"] = 1e-3
            params["Unloading_parameters"]["continuation"] = True
            params["Unloading_parameters"]["method"] = "fixed_point"
            params["Unloading_parameters"]["unload_options"]["maxiter"] = 15
            params["Optimization_parameters"]["passive_maxiter"] = 10
        else:
            params["Optimization_parameters"]["passive_maxiter"] = 100


        params["Patient_parameters"]["fiber_angle_epi"] = fiber_angle_epi
        params["Patient_parameters"]["fiber_angle_endo"] = fiber_angle_endo
        
        params["Patient_parameters"]["mesh_type"] = "lv"
        params["Patient_parameters"]["mesh_group"] = ""
        
        params["Patient_parameters"]["pressure_path"] = os.path.join(filepath,"relative_path_to_pressure_data")
        params["Patient_parameters"]["mesh_path"] = os.path.join(filepath,"relative_path_to_mesh_data")

        
        params["Optimization_parameters"]["gamma_max"] = 1.0
        if c[1] == "active_strain":
            params["T_ref"] = 0.5
            
        else: # Active stress
            params["T_ref"] = 200.0



        for k, v in opttargets.iteritems():
            params["Optimization_targets"][k] = v


        for k, v in optweight_active.iteritems():
            params["Active_optimization_weigths"][k] = v

        for k, v in optweight_passive.iteritems():
            params["Passive_optimization_weigths"][k] = v
            

        params["Material_parameters"] = material_parameters

        params["Patient_parameters"]["patient_type"] = "full"



        outdir = OUTPATH.format(c[0], c[1], c[2])

        # Make directory if it does not allready exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        params["sim_file"] = "/".join([outdir, "result.h5"])

        # Dump paramters to yaml
        with open(fname.format(t), 'wb') as parfile:
            yaml.dump(params, parfile, default_flow_style=False)
        t += 1


    os.system("sbatch run_submit.slurm {} {}".format(t0, t-1))


if __name__ == "__main__":
    main()
