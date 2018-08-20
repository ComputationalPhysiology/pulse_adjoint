"""
This is a regression test
"""
import os
import glob
from pulse_adjoint.setup_parameters import setup_adjoint_contraction_parameters
from pulse_adjoint.example_meshes import mesh_paths, data_paths
from pulse_adjoint.run_full_optimization import main


def test_full_optimization(unload=False):

    params = setup_adjoint_contraction_parameters()
    try:
        os.remove(params["sim_file"])
    except OSError:
        pass

    for f in glob.glob('active_state*'):
        os.remove(f)

    params["log_level"] = 10

    params["Patient_parameters"]["mesh_path"] = mesh_paths['simple_ellipsoid']
    params["Patient_parameters"]["data_path"] = data_paths['unit_data']

    params["Optimization_weigths"]["volume"] = 0.5
    params["Optimization_weigths"]["regional_strain"] = 0.5
    params["Optimization_weigths"]["regularization"] = 0.01

    params["Optimization_parameters"]["passive_maxiter"] = 1
    params["Optimization_parameters"]["active_maxiter"] = 1

    params["unload"] = unload

    if unload:
        params["Unloading_parameters"]["estimate_initial_guess"] = False
        params["Unloading_parameters"]["maxiter"] = 1
        params["Unloading_parameters"]["tol"] = 1e-3
        params["Unloading_parameters"]["continuation"] = False
        params["Unloading_parameters"]["method"] = "raghavan"
        params["Unloading_parameters"]["unload_options"]["maxiter"] = 1
        params["Optimization_parameters"]["passive_maxiter"] = 1
        
    main(params)


if __name__ == "__main__":
    
    test_full_optimization(unload=True)
