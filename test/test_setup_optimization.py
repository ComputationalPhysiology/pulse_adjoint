import pytest

from pulse_adjoint.utils import Object
from pulse_adjoint import setup_optimization2 as so
from pulse_adjoint import setup_parameters
from pulse_adjoint.example_meshes import mesh_paths, data_paths

from test_clinical_data import sample_clinical_data


def test_find_start_end_index():

    data = Object()
    data.passive_duration = 2
    data.num_points = 4

    params = {'phase': 'passive_inflation', 'unload': False}
    start, end = so.find_start_and_end_index(params, data)
    assert start == 0
    assert end == 2

    params = {'phase': 'passive_inflation', 'unload': True}
    start, end = so.find_start_and_end_index(params, data)
    assert start == 0
    assert end == 3

    params = {'phase': 'active_contraction', 'unload': False}
    start, end = so.find_start_and_end_index(params, data)
    assert start == 1
    assert end == 4

    params = {'phase': 'active_contraction', 'unload': True}
    start, end = so.find_start_and_end_index(params, data)
    assert start == 2
    assert end == 5


def test_check_optimization_target_weigths():

    
    from IPython import embed; embed()
    exit()


def test_initialze_patient_data():

    params = setup_parameters.setup_patient_parameters()
    params["mesh_path"] = mesh_paths['simple_ellipsoid']
    params["data_path"] = data_paths['full_data']
    patient = so.initialize_patient_data(params)

    for attr in ['data', 'geometry']:
        assert hasattr(patient, attr)

   
if __name__ == "__main__":
    # test_find_start_end_index()
    # test_check_optimization_target_weigths()
    test_initialze_patient_data()
