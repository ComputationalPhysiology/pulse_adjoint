#!/usr/bin/env python
import os

import dolfin
import dolfin_adjoint
from pulse import numpy_mpi
# from pulse.dolfin_utils import read_hdf5, get_pressure_dict
from pulse.iterate import iterate

from . import config

from .io_utils import get_ed_state_group

# from .utils import logger



class BasicHeartProblem(object):
    """
    This is a basic class for the heart problem.
    """
    def __iter__(self):
        return self

    def __init__(self, bcs, problem):

        self.problem = problem
        self.bcs = bcs
        self.pressure_dict = get_pressure_dict(problem)

        # Create a pressure generator
        self.lv_pressure = (p for p in bcs["pressure"][1:])
        self.pressure_dict['p_lv'].assign(bcs["pressure"][0])
        if 'p_rv' in self.pressure_dict:
            self.pressure_dict['p_rv'].assign(bcs["rv_pressure"][0])
            self.rv_pressure = (p for p in bcs["rv_pressure"][1:])

    def increase_pressure(self):

        p_lv = next(self.lv_pressure)
        if 'p_rv' in self.pressure_dict:
            p_rv = next(self.rv_pressure)
            target_pressure = (p_lv, p_rv)
        else:
            target_pressure = p_lv

        iterate("pressure", self.problem,
                target_pressure, self.pressure_dict,
                continuation=True)

    def get_state(self, copy=True):
        """
        Return a copy of the state
        """
        if copy:
            return self.problem.state.copy(deepcopy=True)
        else:
            return self.problem.state

    def get_gamma(self, copy=True):

        gamma = self.problem.material.activation

        if isinstance(gamma, (dolfin.Constant, dolfin_adjoint.Constant)):
            return gamma

        if copy:
            return gamma.copy(deepcopy=True)
        else:
            return gamma

    def next(self):
        """Solve the system as it is
        """
        self.problem.solve()
        return self.get_state(False)


def get_mean(f):
    return numpy_mpi.gather_broadcast(f.vector().array()).mean()


def get_max(f):
    return numpy_mpi.gather_broadcast(f.vector().array()).max()


def get_max_diff(f1, f2):

    diff = f1.vector() - f2.vector()
    diff.abs()
    return diff.max()


class ActiveHeartProblem(BasicHeartProblem):
    """
    A heart problem for the regional contracting gamma.
    """
    def __init__(self, bcs,
                 problem, params):

        self.acin = params["active_contraction_iteration_number"]
        fname = "active_state_{}.h5".format(self.acin)
        if os.path.isfile(fname):
            if dolfin.mpi_comm_world().rank == 0:
                os.remove(fname)

        BasicHeartProblem.__init__(self, bcs, problem)

        # Load the state from the previous iteration
        w_temp = dolfin.Function(self.problem.state_space)
        # Get previous state
        if self.acin == 0:
            h5group = get_ed_state_group(params["sim_file"],
                                         params["h5group"])

        else:
            h5group \
                = "/".join([params["h5group"], config.
                            ACTIVE_CONTRACTION_GROUP.format(self.acin-1),
                            "states", "0"])

        read_hdf5(h5name=params["sim_file"], func=w_temp, h5group=h5group)

        self.problem.reinit(w_temp)
        self.problem.solve()

    def get_number_of_stored_states(self):

        fname = "active_state_{}.h5".format(self.acin)
        if os.path.isfile(fname):
            i = 0
            with dolfin.HDF5File(dolfin.mpi_comm_world(),
                                 fname, "r") as h5file:
                group_exist = h5file.has_dataset("0")
                while group_exist:
                    i += 1
                    group_exist = h5file.has_dataset(str(i))
            return i

        else:
            return 0


    def store_states(self, states, gammas):

        fname = "active_state_{}.h5".format(self.acin)
        file_mode = "a" if os.path.isfile(fname) else "w"
        key = self.get_number_of_stored_states()

        gamma_group = "{}/gamma"
        state_group = "{}/state"

        assert len(states) == len(gammas), \
            "Number of states does not math number of gammas"

        with dolfin.HDF5File(dolfin.mpi_comm_world(),
                             fname, file_mode) as h5file:

            for (w, g) in zip(states, gammas):
                h5file.write(w, state_group.format(key))
                h5file.write(g, gamma_group.format(key))
                key += 1

    def load_states(self):

        fname = "active_state_{}.h5".format(self.acin)
        if not os.path.isfile(fname):
            return [], []

        nstates = self.get_number_of_stored_states()

        gamma_group = "{}/gamma"
        state_group = "{}/state"

        states = []
        gammas = []

        w = self.problem.state.copy(deepcopy=True)
        g = self.problem.material.activation.copy(deepcopy=True)

        with dolfin.HDF5File(dolfin.mpi_comm_world(),
                             fname, "r") as h5file:

            for i in range(nstates):

                try:
                    h5file.read(w, state_group.format(i))
                    h5file.read(g, gamma_group.format(i))

                except Exception as ex:
                    logger.info(ex)
                    logger.info("State {} does not exist".format(i))

                else:
                    states.append(w.copy(True))
                    gammas.append(g.copy(True))

        return states, gammas

    def next_active(self, gamma_current, gamma,
                    assign_prev_state=True, steps=None):

        old_states, old_gammas = self.load_states()
                
        gammas, states = iterate("gamma", self.problem, gamma_current, gamma,
                                 continuation=True, old_states=old_states,
                                 old_gammas=old_gammas)

        # Store these gammas and states which can be used
        # as initial guess for the newton solver in a later
        # iteration
        self.store_states(states, gammas)

        if assign_prev_state:
            # Assign the previous state
            self.problem.reinit(states[-1])
            self.problem.material.activation.assign(gammas[-1])

        return self.get_state(False)


class PassiveHeartProblem(BasicHeartProblem):
    """
    Runs a biventricular simulation of the diastolic phase of the cardiac
    cycle. The simulation is driven by LV pressures and is quasi-static.
    """
    def next(self):
        """
        Increase the pressure and solve the system
        """
        
        self.increase_pressure()
        
        return self.get_state(False)



