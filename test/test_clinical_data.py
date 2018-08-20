import os
import numpy as np
from pulse_adjoint.clinical_data import ClinicalData


def sample_clinical_data(datasize=4,
                         regions={i for i in range(1, 18)}):

    times = tuple(range(datasize))

    strains = tuple([tuple(0.0 * si for i in range(3))
                     for si in range(datasize)])
    target_strains = {i: strains for i in regions}

    # target_volumes = tuple(np.linspace(100, 150, datasize))
    target_volumes = tuple([27.387594] * datasize)
    target_pressure = tuple(range(0, datasize*2, 2))

    data = ClinicalData(time=times,
                        strain=target_strains,
                        volume=target_volumes,
                        pressure=target_pressure,
                        start_passive=0,
                        start_active=3)

    f = "test.yml"
    # from IPython import embed; embed()
    data.to_yaml(f)
    assert os.path.isfile(f)
    # os.remove(f)

    return data


if __name__ == "__main__":
    sample_clinical_data()
