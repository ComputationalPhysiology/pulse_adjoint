#!/usr/bin/env python
"""
This script contains the container class that loads the results, 
and plot the results easily. 
"""
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
from scipy import stats
import os
import yaml
from ..setup_parameters import setup_general_parameters
from ..dolfin_utils import RegionalParameter
from .args import *

import load, plot, utils, latex_utils, tables, vtk_utils



class PostProcess(object):
    """
    Initialize base class for postprocessing
    
    You need to provide three separate files.
    One HDF file (`fname`) containing the results.
    One HDF file (`geoname`) containing patient data.
    One yaml file (`pname`) containing adjoint contraction paramteters.

    All files should be organized with unque keys. 
    For instace, if you have two patients, e.g CRT and HC
    then two possible keys are 'CRT' and 'HC', and the results
    in `fname` should be orgranised as follows:
    
    ```
    fname
       CRT
          results for CRT
       HC
          results for HC
    ```
    and similarly for the other files.

    Parameters
    ----------
        
    fname: str
        Name of file where the results is stored. 
        See `load.get_data`
    geoname: str
        Name of file where the patient data is stored. 
        See `load.save_patient_to_h5`
    pname: str
        Name of file where the adjoint contraction parameters are stored. 
        See `load.save_parameters`
    outdir: str
        Directory where to save the results
    recompute: bool
        The data is stored in a yaml file and 
        will be loaded if it exist and recompute is False. 
        If recompute is True, then the feature will be recomputed
    
    """
    
    def __init__(self, fname,
                 geoname, 
                 pname,
                 outdir,
                 tmp_dir = None,
                 recompute = False):
        

        logger.info("Load file {}".format(fname))
        
        self._load_data(fname)
        self._fname = fname
        self._geoname = geoname
        self._pname = pname
        self.set_feature_keys()
        
        self._recompute = recompute
        self._outdir = outdir
        self._keys = self._data.keys()
        # self._mesh_path = mesh_path
        # self._pressure_path = pressure_path
        # self._echo_path = echo_path
        # self._params = params
        # self._simdir = simdir

        if tmp_dir is None:
            self._tmp_resdir = "/".join([os.path.abspath(os.path.dirname(fname)), "tmp_results"])
        else:
             self._tmp_resdir = tmp_dir
            
        if not os.path.exists(self._tmp_resdir):
            os.makedirs(self._tmp_resdir)
            
        
        self._results = {}
        self._features = {}


    def _load_data(self, fname):
        """Load data

        :param fname: path to datafile

        """
        
        self._data = load.load_dict_from_h5(fname)
        

    def _data_exist(self, patient_name, key):

        name = "_".join([patient_name, key]) + ".yml"
        return os.path.isfile("/".join([self._tmp_resdir, name]))

    def _load_tmp_results(self, patient_name, key, d):

        name = "_".join([patient_name, key]) + ".yml"
        path = "/".join([self._tmp_resdir, name])
        
        with open(path, 'rb') as parfile:
            data = yaml.load(parfile)

        d.update(**data)

    def _save_tmp_results(self, patient_name, key, d):
        
        name = "_".join([patient_name, key]) + ".yml"
        path = "/".join([self._tmp_resdir, name])

        with open(path, 'wb') as f:
            yaml.dump(d, f, default_flow_style=False)

    def _update_results(self, data):
        """ Update results
        """

        if len(data.keys()) == 0:
            return
        
        if len(self._results.keys()) == 0:
            self._results.update(**data)
        else:
            utils.update_nested_dict(self._results, data)

    def _update_features(self, data):
        """ Update results
        """
        if len(self._features.keys()) == 0:
            self._features.update(**data)
        else:
            utils.update_nested_dict(self._features, data)

    def _set_matparams(self, patient, patient_name):

        params = self._params

        matlst = params["Fixed_parameters"].keys()


        if params["matparams_space"] == "regional":
            sfun = merge_control(patient, params["merge_passive_control"])
            paramvec = RegionalParameter(sfun)
        else:
            family, degree = params["matparams_space"].split("_")                
            paramvec = dolfin.Function(dolfin.FunctionSpace(patient.mesh, family, int(degree)), name = "matparam vector")

        fixed_idx = np.nonzero([not params["Fixed_parameters"][k] for k in matlst])[0][0]
        par = matlst[fixed_idx]

        if self._data[patient_name].has_key("material_parameters"):
            paramvec.vector()[:] = self._data[patient_name]["material_parameters"][par]
        
        
            matparams = {k:v[0] for k,v in self._data[patient_name]["material_parameters"].iteritems()}
            matparams[par] = paramvec
            
            if not isinstance(params, dict):
                params = params.to_dict()
            params["Material_parameters"].update(**matparams)
            
        self._params = params
        return params

    def get_average_valve_times(self):
        """Get the percentage  average value of the time
        of the different valvular events starting at mvc

        Example

          mvc: 0%, avo: 5% , avc: 40%, mvo: 50%, 

        :returns: 
        :rtype: 

        """
        

        if self._valve_times is None:
            return None
        
        avo, avc, mvo = [], [], []
        
        for patient_name, v in self._valve_times.iteritems():
            if not v.has_key("num_points"): continue
            n = v["num_points"]
            echo_valve_times = v#["echo_valve_time"]
            pfb = v["passive_filling_begins"]

           
            avo.append(((echo_valve_times["avo"]-echo_valve_times["mvc"])%n)/float(n))
            avc.append(((echo_valve_times["avc"]-echo_valve_times["mvc"])%n)/float(n))
            mvo.append(((echo_valve_times["mvo"]-echo_valve_times["mvc"])%n)/float(n))
            

        d = {"mvc":0.0,
             "avo": np.mean(avo),
             "avc": np.mean(avc),
             "mvo": np.mean(mvo)}
        return d

    def _setup_compute(self, patient_name):

        self._label_key = r"Patient"
        
        # params = self._params
        # simdir = self._simdir
        # mesh_path = self._mesh_path
        # pressure_path = self._pressure_path
        # echo_path = self._echo_path

        logger.info("\nProcess data for patient {}".format(patient_name))

        self._params = load.load_parameters(self._pname, patient_name)
        patient = load.load_patient_data(self._geoname, patient_name)        
        params = self._set_matparams(patient, patient_name)
        patient = load.load_measured_strain_and_volume(patient, params)

        from pulse_adjoint.setup_optimization import check_patient_attributes
        check_patient_attributes(patient)

         
        return params, patient 


    def get_data(self, patient_name):

        val = self._data[patient_name]

        for k in ["displacements", "gammas", "states"]:
            if not val.has_key(k):
                val[k] = {}

       
                 
        return val

    def get_meshvolumes(self, patient_name):
        """
        Return a dictionary with the volumes of each segement
        in the mesh.

        Parameters
        ----------

        patient_name : str
            The key/name of patient in the results dictionary.

        Returns
        -------

        meshvols : dict
             A dictionary with the volumes of each segement in the mesh.

        """
        patient = load.load_patient_data(self._geoname, patient_name)
     
        dx = dolfin.Measure("dx", domain=patient.mesh, subdomain_data=patient.sfun)
        meshvols = {}
        for r in set(patient.sfun.array()):
            meshvols[r] = dolfin.assemble(dolfin.Constant(1.0)*dx(int(r)))

        return meshvols

    
    def compute(self, *args):
        """
        Compute features that can be plotted later.
        Possible inputs are:

        * 'volume'
        * 'strain'
        * 'mean_gamma'
        * 'regional_gamma'
        * 'measurements'
        * 'emax'
        * 'geometric_distance'
        * 'time_varying_elastance'
        * 'vtk_simulation'
        * 'cardiac_work:{case}:{work_pair}', where case is 
           "full", "comp_fiber" or "comp_long", and work_pair 
           is "SE", "SEdev" "PF" or "pgradu"

        """

        if len(args) == 0:
            logger.info("Nothing to compute. Return")
            return
        
        setup_general_parameters()
        
        
        data = {}
        self._passive_filling_duration = {}
        for patient_name in self._keys:

            
            val = self.get_data(patient_name)
            

            
            data[patient_name] = {}

            params, patient = self._setup_compute(patient_name)
            self._passive_filling_duration[patient_name] = patient.passive_filling_duration
            
            def compute_echo_work(patient, params):
                strain = {}
                load.load_measured_strain(strain, patient, "measured_strain")
                try:
                    # Not all the patients have meausred work
                    measured_work \
                        ={k:np.transpose(v)[2].tolist() for k, v \
                          in patient.work.iteritems()}
                except:
                    # Compute work as how it would be measured from the strains
                    measured_work = {}
                    pressure = patient.pressure[1:] if params["unload"] \
                               else patient.pressure
                    for r, s in strain["measured_strain"]["longitudinal"].iteritems():


                        measured_work[r] =utils.compute_cardiac_work_echo(pressure,
                                                                          s, flip =True)

                return measured_work
                        
            if "measurements" in args:

                
                d = {}
                
                d["measured_volume"] = patient.volume
                d["pressure"] = patient.pressure                 
                
                load.load_measured_strain(d, patient, "measured_strain")

                if hasattr(patient,"rv_volume"):
                    d["measured_volume_rv"] = patient.rv_volume #\
                                              # + patient["data"]["rv_volume_offset"]
                    d["rv_pressure"] = patient.rv_pressure# \
                                       # + patient["data"]["rv_pressure_offset"]


                d["measured_work"] = compute_echo_work(patient, params)
                data[patient_name].update(**d)
                    
    
            if "volume" in args:

               
                if self._data_exist(patient_name, "volume") and not self._recompute:
                    self._load_tmp_results(patient_name, "volume", data[patient_name])

                else:
                    logger.info("Compute volume")
                
                    d = {}
                    
                    # LV
                    d["measured_volume"] = patient.volume
                    d["simulated_volume"] = utils.get_volumes(val["displacements"],
                                                              patient, "lv",
                                                              self._params["volume_approx"])
                    d["pressure"] = patient.pressure
                    
                    # RV
                    if hasattr(patient, "rv_volume"):
                         d["measured_volume_rv"] = patient.rv_volume
                         
                         d["simulated_volume_rv"] = utils.get_volumes(val["displacements"],
                                                                      patient, "rv",
                                                                      self._params["volume_approx"])

                         d["rv_pressure"] = patient.rv_pressure


                    for k, v in d.items():
                        if isinstance(v, np.ndarray):
                            d[k] = v.tolist()
                            
                    self._save_tmp_results(patient_name, "volume", d)
                    data[patient_name].update(**d)


            if "strain" in args:
                if self._data_exist(patient_name, "strain") and not self._recompute:
                    self._load_tmp_results(patient_name, "strain", data[patient_name])

                else:
                   
                    logger.info("Compute strain")
                    d = {}
                    d["simulated_strain"] \
                        = utils.get_regional_strains(val["displacements"],
                                                     patient,**params)

                  
                    
                    load.load_measured_strain(d, patient, "measured_strain")
                   
                    self._save_tmp_results(patient_name, "strain", d)
                    data[patient_name].update(**d)

            if "almansi_strain" in args:
                
                if self._data_exist(patient_name, "almansi_strain") and not self._recompute:
                    self._load_tmp_results(patient_name, "almansi_strain", data[patient_name])

                else:
                   
                    logger.info("Compute almansi strain")
                    d = {}
                    d["almansi_strain"] \
                        = utils.get_regional_strains(val["displacements"],
                                                     patient,almansi=True,**params)
                    
                  
                                       
                    self._save_tmp_results(patient_name, "almansi_strain", d)
                    data[patient_name].update(**d)

                    
            if 'time_varying_elastance' in args:

                if self._data_exist(patient_name, 'time_varying_elastance') and not self._recompute:
                    self._load_tmp_results(patient_name, 'time_varying_elastance', data[patient_name])

                else:
                    logger.info("Compute time_varying_elastance")
                    d = utils.compute_time_varying_elastance(patient, params, val)
                    self._save_tmp_results(patient_name, 'time_varying_elastance', d)
                    data[patient_name].update(**d)
                         

            if "emax" in args:

                if self._data_exist(patient_name, "emax") and not self._recompute:
                    self._load_tmp_results(patient_name, "emax", data[patient_name])


                else:
                    logger.info("Compute emax")
                    
                    d = {"emax": utils.compute_emax(patient,
                                                    params,
                                                    val,
                                                    self._valve_times[patient_name])}
                    
                    self._save_tmp_results(patient_name, "emax", d)
                    data[patient_name].update(**d)

            if "end_systolic_elastance" in args:

                if self._data_exist(patient_name, "end_systolic_elastance") and not self._recompute:
                    self._load_tmp_results(patient_name, "end_systolic_elastance", data[patient_name])


                else:
                    logger.info("Compute end_systolic_elastance")

                    es = self._es[patient_name]
                    

                    state = val["states"][str(es)]
                    gamma = val["gammas"][str(es)]

                    matparams = val["material_parameters"]
                    
                    if params["matparams_space"] == "regional":
                        sfun = merge_control(patient, params["merge_passive_control"])
                        mat = RegionalParameter(sfun)
                        mat.vector()[:] = matparams["a"]
                    else:
                        family, degree = params["matparams_space"].split("_")
                        mat_space= dolfin.FunctionSpace(moving_mesh, family, int(degree))
                        mat = dolfin.Function(mat_space, name = "material_parameter_a")
                        mat.vector()[:] = matparams["a"]

                    matparams["a"] = mat

                    for k in ["a_f", "b", "b_f"]:
                        v = dolfin.Constant(matparams[k][0])
                        matparams[k] = v


                    d = {}
                    if patient.is_biv():
                        pressure = (patient.pressure[es],patient.rv_pressure[es])
                        
                        d["end_systolic_elastance_rv"] =  utils.compute_elastance(state, pressure, gamma, patient,
                                                                                  params, matparams,chamber="rv")
                        
                    else:
                        pressure = patient.pressure[es]
                        
                    

                    d["end_systolic_elastance"] = utils.compute_elastance(state, pressure, gamma, patient,
                                                                           params, matparams,chamber="lv")

                    self._save_tmp_results(patient_name, "end_systolic_elastance", d)
                    data[patient_name].update(**d)
                    
                
            if "gamma_mean" or "gamma_regional" in args:
                
                dX = dolfin.Measure("dx",subdomain_data = patient.sfun,
                     domain = patient.mesh)
                regions = [int(r) for r in set(patient.sfun.array())]

                gs = val["gammas"]
                gamma_lst = [gs[k] for k in sorted(gs, key=utils.asint)]

                if params["gamma_space"] == "regional":
                    sfun = merge_control(patient, params["merge_active_control"])
                    gamma = RegionalParameter(sfun)
                else:
                    gamma_space = dolfin.FunctionSpace(patient.mesh, "CG", 1)
                    gamma = dolfin.Function(gamma_space, name = "Contraction Parameter")

             
                if "gamma_mean" in args:
                    if self._data_exist(patient_name, "gamma_mean") and not self._recompute:
                        self._load_tmp_results(patient_name, "gamma_mean", data[patient_name])

                    else:
                        logger.info("Compute mean gamma")
                        d = {"gamma_mean": utils.get_global(dX, gamma, gamma_lst, regions,params["T_ref"])}
                        self._save_tmp_results(patient_name, "gamma_mean", d)
                        data[patient_name].update(**d)
                
                if "gamma_regional" in args:
                    if self._data_exist(patient_name, "gamma_regional") and not self._recompute:
                        self._load_tmp_results(patient_name, "gamma_regional", data[patient_name])

                    else:
                        logger.info("Compute regional gamma")
                        d = {"gamma_regional": utils.get_regional(dX, gamma,  gamma_lst, regions, params["T_ref"])}
                        self._save_tmp_results(patient_name, "gamma_regional", d)
                        data[patient_name].update(**d)

                        
                        
            if "geometric_distance" in args:

                if self._data_exist(patient_name, "geometric_distance"):
                    self._load_tmp_results(patient_name, "geometric_distance", data[patient_name])
                else:
                    logger.info("Compute geometric distance")
                    vtk_output = "/".join([self._outdir, "surface_files2", patient_name])
                    d = utils.compute_geometric_distance(patient, val["displacements"], vtk_output)
                    self._save_tmp_results(patient_name, "geometric_distance", d)
                    data[patient_name].update(**d)

            if "data_mismatch" in args:

                if self._data_exist(patient_name, "data_mismatch"):
                    self._load_tmp_results(patient_name, "data_mismatch", data[patient_name])
                else:
                    logger.info("Compute data mismatch")
                    d = utils.copmute_data_mismatch(val["displacements"],
                                                    patient["geometry"],
                                                    patient["data"]["volume"],
                                                    patient["data"]["strain"])  
                    
                    self._save_tmp_results(patient_name, "data_mismatch", d)
                    data[patient_name].update(**d)

            if "vtk_simulation" in args or "mechanical_features" in args:

                output = "/".join([os.path.dirname(self._geoname), "features", patient_name + ".h5"])


                if not os.path.isfile(output) or self._recompute:
                    logger.info("Compute meachanical features")
                    d = {"features_scalar" : utils.copmute_mechanical_features(patient,
                                                                               params,
                                                                               val, output,
                                                                               keys = self._feature_keys)}
                    self._save_tmp_results(patient_name, "features_scalar", d)

                self._load_tmp_results(patient_name,
                                       "features_scalar",
                                       data[patient_name])
                features = {patient_name: load.load_dict_from_h5(output)}

                self._update_features(features)

            if  "mechanical_features_scalar" in args:

                output = "/".join([os.path.dirname(self._geoname),
                                   "features", patient_name + ".h5"])

                if not os.path.isfile(output) or self._recompute:
                    logger.info("Compute meachanical features")
                    d = {"features_scalar" : utils.copmute_mechanical_features(patient,
                                                                               params,
                                                                               val, output,
                                                                               keys = self._feature_keys)}
                    self._save_tmp_results(patient_name, "features_scalar", d)

                self._load_tmp_results(patient_name,
                                       "features_scalar",
                                       data[patient_name])

                
            if any([a.startswith("cardiac_work") for a in args]):

                try:
                    idx = np.where([a.startswith("cardiac_work") for a in args])[0][0]
                    string = args[idx]
                    _,  case, wp = string.split(":")

                except:
                    string = 'cardiac_work:comp_long:SE'
                    _,  case, wp = string.split(":")
                    

               
                assert wp in work_pairs, "Illegal work pair: {}. Legal inputs:{}".format(wp,work_pairs)
                # assert case in cases, "Illegal case: {}. Legal inputs:{}".format(case, cases)
                

                if self._data_exist(patient_name, string) and not self._recompute:
                    self._load_tmp_results(patient_name, string, data[patient_name])
                    
                else:

                    if 0:#string == 'cardiac_work:comp_long:pgradu':

                        strain = utils.get_regional_strains(val["displacements"],
                                                            patient,**params)
                        work = {}
                        pressure = patient.pressure[1:] if params["unload"] \
                                   else patient.pressure
                        for r, s_ in strain["longitudinal"].iteritems():

                            s = s_[1:] if params["unload"] else s_
                            work[r] = utils.compute_cardiac_work_echo(pressure,
                                                                      s, flip =True)
                            
                        data[patient_name]["work_pgradu_comp_long"] = work
                    else:


                       
                        logger.info("Compute cardiac work")
                        d = utils.compute_cardiac_work(patient, params, val, case, wp)
                    
                        self._save_tmp_results(patient_name, string, d)
                        data[patient_name].update(**d)

                data[patient_name]["measured_work"]  = compute_echo_work(patient, params)
                # data[patient_name]["measured_work"] \
                    # ={k:np.transpose(v)[2].tolist() for k, v in patient.work.iteritems()}
                    

        self._update_results(data)

    def get_solver(self, patient_name, idx=0):

        params, patient = self._setup_compute(patient_name)
        # gamma = self._
        
  
    def plot_pv_loop(self, single=True, groups = None, chamber = "lv"):
        """Plot pressure volume loop(s)

        :param path: Path to where to save the figures
        :param single: If true, plot single pvloops else plot them all together

        """

        
        # Make sure that we have computed the volumes
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("simulated_volume"):
            self.compute("volume")


        outdir = "/".join([self._outdir, "pv_loop2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        logger.info("Plot PV loops")
        
            
        if single:
            # Plot a separate figure for each patient
            
            for patient_name in self._results.keys():

                params, patient  = self._setup_compute(patient_name)
                
                v_sim = self._results[patient_name]["simulated_volume"]
                v_meas = self._results[patient_name]["measured_volume"]
                pressure = self._results[patient_name]["pressure"]
                path = "/".join([outdir, patient_name])

                plot.plot_single_pv_loop(v_sim, v_meas, pressure, path, params["unload"])

                if self._results[patient_name].has_key("measured_volume_rv"):
                    v_sim = self._results[patient_name]["simulated_volume_rv"]
                    v_meas = self._results[patient_name]["measured_volume_rv"]
                    pressure = self._results[patient_name]["rv_pressure"]
                    path = "/".join([outdir, patient_name + "_rv"])

                    plot.plot_single_pv_loop(v_sim, v_meas, pressure, path, params["unload"])

        else:
            # Plot all PV loops in the same figure
            vs_sim, vs_meas, pressures, labels = [], [], [], []
            
            for patient_name in self._results.keys():

                vs_sim.append(self._results[patient_name]["simulated_volume"])
                vs_meas.append(self._results[patient_name]["measured_volume"])
                pressures.append(self._results[patient_name]["pressure"])
                if groups:
                    labels.append(groups[patient_name])
                else:
                    labels.append(patient_name)
                
            path = "/".join([outdir, "all_loops"])    
            plot.plot_multiple_pv_loop(vs_sim, vs_meas, pressures, path, labels)

    def plot_pv_loop_w_elastance(self, single=True, groups = None):
        """Plot pressure volume loop(s)

        :param path: Path to where to save the figures
        :param single: If true, plot single pvloops else plot them all together

        """

        
        # Make sure that we have computed the volumes
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("simulated_volume"):
            self.compute("volume")

        if len(keys) == 0 or not self._results[keys[0]].has_key("elastance"):
            self.compute("time_varying_elastance")

        outdir = "/".join([self._outdir, "pv_loop_elastance"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        logger.info("Plot PV loops with elastance ")

        if single:
            # Plot a separate figure for each patient
            
            for patient_name in self._results.keys():

                v_sim = self._results[patient_name]["simulated_volume"]
                v_meas = self._results[patient_name]["measured_volume"]
                pressure = self._results[patient_name]["pressure"]
                es = self._results[patient_name]["elastance"]
                v0s = self._results[patient_name]["v0"]
                ES = self._valve_times[patient_name]["echo_valve_time"]["avc"]
                path = "/".join([outdir, patient_name])
                
                plot.plot_pv_loop_w_elastance([v_sim], [v_meas], [pressure],
                                              [es],[v0s], [ES], path,[""])

        else:
            # Plot all PV loops in the same figure
            ES, es, v0s,  vs_sim, vs_meas, pressures, labels = [], [], [], [], [], [], []
            
            for patient_name in self._results.keys():

                vs_sim.append(self._results[patient_name]["simulated_volume"])
                vs_meas.append(self._results[patient_name]["measured_volume"])
                pressures.append(self._results[patient_name]["pressure"])
                es.append(self._results[patient_name]["elastance"])
                v0s.append(self._results[patient_name]["v0"])
                ES.append(self._valve_times[patient_name]["echo_valve_time"]["avc"])
                if groups:
                    labels.append(groups[patient_name])
                else:
                    labels.append(patient_name)
                
            path = "/".join([outdir, "all_loops"])    
            plot.plot_pv_loop_w_elastance(vs_sim, vs_meas, pressures,
                                          es, v0s, ES, path, labels)



    def plot_strain_curves(self, region = 1, component = "longitudinal", groups = None):
        
        # Make sure that we have computed the strains
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("simulated_strain"):
            self.compute("strain")

        outdir = "/".join([self._outdir, "strain2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # N = 100
        # valve_times = {"passive_filling_begins":0,
        #                "echo_valve_time":{"mvc": 0, "avo": int(3*N*float(0.05)),
        #                                   "avc":int(3*N*float(0.4)), "mvo":int(3*N*float(0.50))}}

        simulated, measured = {},{}
        for patient_name in self._results.keys():

            simulated[patient_name] = self._results[patient_name]["simulated_strain"][component][region]
            
            measured[patient_name] = self._results[patient_name]["measured_strain"][component][region]
                                     

        path = "/".join([outdir, "strain_curves"])
        plot.plot_single_strain_curves(simulated, measured, path, groups, self._params["unload"])

        
            
    def plot_strain(self, dirs = ['circumferential','radial', 'longitudinal'], nregions = 17):
        """Plot all strain curves

        """
        
        # Make sure that we have computed the strains
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("simulated_strain"):
            self.compute("strain")


        outdir = "/".join([self._outdir, "strain2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        logger.info("Plot strain curves")

        
        for patient_name in self._results.keys():

            
            params, patient  = self._setup_compute(patient_name)

            simulated = self._results[patient_name]["simulated_strain"]
            measured =  self._results[patient_name]["measured_strain"]
            paths = plot.plot_strains(simulated, measured, outdir, dirs, nregions, params["unload"])
            
            if len(dirs) == 3:
                latex_utils.make_canvas_strain(paths, patient_name)

    def print_timings(self):

        timings = {k:[] for k in self._data[self._keys[0]]["timings"].keys()}
        ncontrols = []
        
        for patient_name in self._keys:

            ncontrols.append(len(self._data[patient_name]["gammas"]["0"]))
            for k,v in self._data[patient_name]["timings"].iteritems():
                
                timings[k].append(v)

        

        forward_times = np.concatenate([f.tolist() for t \
                                        in timings["forward_times"] \
                                        for f in t.values()])

        print("\nForward times = {} +/- {}".format(np.mean(forward_times),
                                                 np.std(forward_times)))
        
        backward_times = np.concatenate([f.tolist() for t \
                                        in timings["backward_times"] \
                                        for f in t.values()])

        print("\nBackward times = {} +/- {}".format(np.mean(backward_times),
                                                 np.std(backward_times)))

        print("\nNumber of controls = {} +/- {}".format(np.mean(ncontrols),
                                                      np.std(ncontrols)))

        run_times = np.concatenate([f.tolist() for f \
                                    in timings["run_time"]])

        
        print("\nRun time = {} +/- {}".format(np.mean(run_times),
                                              np.std(run_times)))

            
            
    def plot_strain_scatter(self, split = 1,  active_only = False, unload = True):
        """
        Plot all the strain data in a scatter plot, 
        with measured on the x-axis and simulated on the 
        y-axis. If the fit is good, then they should lie 
        on the straight line. 

        :param int split: Color the dots from passive and active phase in 
                          different colors. Possible values: [1,2,3]
        :param bool active_only: If true, exclude the points from the passive 
                                 parameter fitting
        """
        outdir = "/".join([self._outdir, "strain2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Make sure that we have computed the strains
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("simulated_strain"):
            self.compute("strain")

        logger.info("Plot strain scatter plot")
        
        start = 1 if unload else 0

        from copy import deepcopy
        strain_comp = {"longitudinal":[],
                       "circumferential":[],
                       "radial":[]}
        data = {"simulated_active":deepcopy(strain_comp),
                "simulated_passive":deepcopy(strain_comp),
                "measured_active":deepcopy(strain_comp),
                "measured_passive":deepcopy(strain_comp)}
        
        for patient_name in self._results.keys():

            sim =  self._results[patient_name]["simulated_strain"]
            meas = self._results[patient_name]["measured_strain"]
            pfd = self._valve_times[patient_name]["passive_filling_duration"]
            
            for direction in sim.keys():
                for region in sim[direction].keys():

                    if not np.all(np.array(meas[direction][region]) == 0.0):

                        # print "Patient: {}, Simulated: {}, Measured: {}".format(patient_name,
                        #                                                         len(sim[direction][region]),
                        #                                                         len(meas[direction][region]))
                        arr_sim = np.array(sim[direction][region][start+pfd:]).tolist()
                        data["simulated_active"][direction] += arr_sim
                        data["measured_active"][direction] += np.array(meas[direction][region][pfd:pfd+len(arr_sim)]).tolist()

                        data["simulated_passive"][direction] += np.array(sim[direction][region][start:pfd+start]).tolist()
                        data["measured_passive"][direction] += np.array(meas[direction][region][:pfd]).tolist()
                        

      
        if split == 0:
            # Plot everything together
            
            simulated = []
            measured = []
            new_data = {"simulated":[], "measured":[]}
            
            for phase in ["active", "passive"]:
                for comp in strain_comp.keys():
                    new_data["simulated"] += data["simulated_{}".format(phase)][comp]
                    new_data["measured"] += data["measured_{}".format(phase)][comp]

            
            path = "/".join([outdir,"strain_scatter_0"])
            plot.plot_strain_scatter(new_data, path, split = 0)
            
        elif split == 1:
            # Split into passive and active

            new_data = {"simulated_active":[],
                        "simulated_passive":[],
                        "measured_active":[],
                        "measured_passive":[]}

            for key in new_data.keys():
                for comp in strain_comp.keys():
                    new_data[key] += data[key][comp]


            labels = {"passive": "diastolic",
                      "active": "systolic"}
            path = "/".join([outdir,"strain_scatter_1"])
            plot.plot_strain_scatter(new_data,path, labels, split = 1)
        else:
            # Split into active and passive, and the different components
            path = "/".join([outdir,"strain_scatter_2"])
            plot.plot_strain_scatter(data,path, split = 2)
                                 
                        
    def plot_volume_scatter(self, unload=True):
        """
        Plot all the volume data in a scatter plot, 
        with measured on the x-axis and simulated on the 
        y-axis. If the fit is good, then they should lie 
        on the straight line. 
        """
        
        outdir = "/".join([self._outdir, "volume2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Make sure that we have computed the volume
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("simulated_volume"):
            self.compute("volume")

        logger.info("Plot volume scatter plot")

        simulated = []
        measured = []

        start = 1 if unload else 0

        for patient_name in self._results.keys():

            arr = np.array(self._results[patient_name]["simulated_volume"]).tolist()[start:]
            simulated += arr

            end = start + len(arr) 
            measured += np.array(self._results[patient_name]["measured_volume"]).tolist()[start:end]

        
            print "Patient:{}, Simulated: {}, Measured: {}".format(patient_name, len(simulated), len(measured))

        path = "/".join([outdir,"volume_scatter"])
        plot.plot_volume_scatter(simulated, measured, path)
        

    def plot_gamma(self, data="both"):
        """
        Plot gamma curves

        :param str data: What to plot: 'mean', 'regional' or 'both'.
                         Regional curves will be plotted in the same figure, 
                         and it both is selected then both the global and 
                         the regional curves will be plotted together. 

        """

        datalst = ["mean", "regional", "both"]
        msg = "Wrong input for varible data. " +\
              "Possible values are {}. ".format(datalst) +\
              "Given data was {}".format(data)
        assert data in datalst, msg
        

        outdir = "/".join([self._outdir, "gamma2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        logger.info("Plot gamma")

        # Make sure that we have computed the mean/regional gamma
        keys = self._results.keys()
        if data == "both" or "mean":
            if len(keys) == 0 or not self._results[keys[0]].has_key("gamma_mean"):
                self.compute("gamma_mean")
        
        if data == "both" or "regional":
            if len(keys) == 0 or not self._results[keys[0]].has_key("gamma_regional"):
                self.compute("gamma_regional")
            

        for patient_name in self._results.keys():
     
            valve_times = None if not hasattr(self, '_valve_times') else self._valve_times[patient_name]
      
            if data == "mean" or "both":
                path = "/".join([outdir, patient_name + "_mean"])
                plot.plot_single_mean_gamma(np.transpose(self._results[patient_name]["gamma_mean"]),
                                            path, valve_times)
                    
            if data == "regional" or "both":
                path = "/".join([outdir, patient_name + "_regional"])
                plot.plot_single_regional_gamma(np.transpose(self._results[patient_name]["gamma_regional"]),
                                                path, valve_times,
                                                include_global = (data =="both"))




    def plot_gamma_synch(self, data="both", groups = None):
        """Plot gamma curves synchronized to valvular events.
        
        .. note::

           Valvular events has to be parsed in the initialization of this class.

        :param str data: What to plot: 'mean', 'regional' or 'both'. 
                         Each region will be plotted in a separate figure
        :param dict groups: A dictionary with list patient names. 
                            For instance [healthy, HF]. if single = False, then only the 
                            individual patients in each group with be plotted together. 

        """

        # Make sure that we have computed the mean/regional gamma
        keys = self._results.keys()
    
        if data in ("both", "mean"):
            if len(keys) == 0 or not self._results[keys[0]].has_key("gamma_mean"):
                self.compute("gamma_mean")
        
        if data in ("both", "regional"):
            if len(keys) == 0 or not self._results[keys[0]].has_key("gamma_regional"):
                self.compute("gamma_regional")

        logger.info("Plot sychronized gamma (individual)")
        
        outdir = "/".join([self._outdir, "gamma2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        

        # Length of each chunk
        N = 100
        valve_times = {"passive_filling_begins":0,
                       "echo_valve_time": {"mvc":0, "avo":100, "avc":200, "mvo":300}}

        def interpolate_data(patient_name, mean, regional, labels):

            labels.append(" ".join(patient_name.split("_")))
            
            if data in ("mean", "both"):

                f = utils.interpolate_trace_to_valve_times(self._results[patient_name]["gamma_mean"],
                                                           self._valve_times[patient_name], N)
                mean.append(f)

            if data in ("regional",  "both"):

                gammas = []
                for g in self._results[patient_name]["gamma_regional"]:
                    f = utils.interpolate_trace_to_valve_times(g, self._valve_times[patient_name], N)
                    gammas.append(f)
                    
                regional.append(gammas)

        def plot_data(mean, regional, labels, name = ""):
            if data in ("mean", "both"):
                path = "/".join([outdir,"{}global_synch".format(name)])
                plot.plot_multiple_gamma(mean, path, labels, valve_times)

            if data in ("regional", "both"):
                logger.warning("Regional plotting is not yet working")
                # for i,g in enumerate(np.reshape(regional, (-1, len(labels), 4*N))):
                    # path = "/".join([outdir,"{}region_{}_synch".format(name, i+1)])
                    # plot.plot_multiple_gamma(g, path, labels, valve_times)



        if groups is None:
            mean, regional, labels  = [], [], []
            
            for patient_name in self._results.keys():
                interpolate_data(patient_name, mean, regional, labels)

            plot_data(mean, regional, labels)
        else:
            for k, v in groups.iteritems():
                mean, regional, labels  = [], [], []
                for patient_name in v:
                    if patient_name in self._results.keys():
                        interpolate_data(patient_name, mean, regional, labels)
                    else:
                        logger.warning("Patient {} not found".format(patient_name))

                plot_data(mean, regional, labels, k+"_")

       

    def plot_gamma_synch_mean(self, data="both", groups = None):
        """Plot mean gamma curves and standard deviation synchronized to valvular events.
        
        .. note::

           Valvular events has to be parsed in the initialization of this class.

        :param data: What to plot: 'mean', 'regional' or 'both'. 
                     Each region will be plotted in a separate figure
        :param groups: A dictionary with list patient names.
                       For instance [healthy, HF]. if single = False, then only the individual 
                       patients in each group with be plotted together. 

        """
  
        # Make sure that we have computed the mean/regional gamma
        keys = self._results.keys()
        print keys
        if data in ("both", "mean"):
            if len(keys) == 0 or not self._results[keys[0]].has_key("gamma_mean"):
                self.compute("gamma_mean")
        
        if data in ("both", "regional"):
            if len(keys) == 0 or not self._results[keys[0]].has_key("gamma_regional"):
                self.compute("gamma_regional")


        logger.info("Plot sychronized gamma (mean/std)")

        
        outdir = "/".join([self._outdir, "gamma2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            

        # Length of each chunk
        N = 100
        valve_times = {"passive_filling_begins":0,"mvc": 0, "avo": int(3*N*float(0.05)),
                                         "avc":int(3*N*float(0.4)), "mvo":int(3*N*float(0.50))}
                       # "echo_valve_time": {"mvc":0, "avo":N, "avc":2*N, "mvo":3*N}}

        def interpolate_data(patient_name, mean, regional):

            
            if data in ("mean", "both"):

                f = utils.interpolate_trace_to_valve_times(self._results[patient_name]["gamma_mean"],
                                                           self._valve_times[patient_name], N)
                mean.append(f)

            if data in ("regional", "both"):

                gammas = []
                for g in self._results[patient_name]["gamma_regional"]:
                    f = utils.interpolate_trace_to_valve_times(g, self._valve_times[patient_name], N)
                    gammas.append(f)
                    
                regional.append(gammas)

        def plot_data(mean, regional, labels):
            
            if data in ("mean", "both"):

                means = []
                stds = []
                for m in mean:
                    means.append(np.mean(m, 0))
                    stds.append(np.std(m, 0))
                    
                path = "/".join([outdir,"global_synch_mean"])
                plot.plot_gamma_mean_std(means, stds, path, labels, valve_times)

            if data in ("regional" or "both"):

                logger.warning("Regional not yet implemeted")
                # means = []
                # stds = []
                # for m in mean:
                #     means.append(np.mean(m, 0))
                #     stds.append(np.std(m, 0))
                # for i,g in enumerate(np.reshape(regional, (-1, len(labels), 4*N))):
                #     path = "/".join([outdir,"region_{}_synch_mean".format(name, i+1)])
                #     plot.plot_multiple_gamma(g, path, labels, valve_times)


        

        if groups is None:
            mean, regional  = [], []
            
            for patient_name in self._results.keys():
                interpolate_data(patient_name, mean, regional)

            plot_data([mean], [regional], [""])
        else:

            mean, regional = [], []
            for k, v in groups.iteritems():
                mean_group, regional_group  = [], []
                for patient_name in v:
                    if patient_name in self._results.keys():
                        interpolate_data(patient_name, mean_group, regional_group)
                    else:
                        logger.warning("Patient {} not found".format(patient_name))
                mean.append(mean_group)
                regional.append(regional_group)


            # Do some statistics on the groups
            msg = "#"*40 + "\nGamma traces\n" + \
                  "Testing if the means of the groups \n{} ".format(groups.keys()) + \
                  "\nare equal using one-way ANOVA for peak and time-to-peak:\n"
            logger.info(msg)

            # Peak value
            peak = [np.max(m, 1) for m in mean]
            F,p = stats.f_oneway(*peak)
            logger.info("\tPeak: P-value = {}".format(p))

            # Time to peak
            ttpeak = [np.argmax(m, 1) for m in mean]
            F,p = stats.f_oneway(*ttpeak)
            logger.info("\tTime-to-peak: P-value = {}\n".format(p))
            
            logger.info("#"*40)

            plot_data(mean, regional, groups.keys())

    def plot_emax(self, groups = None):
        """
        Plot a bar plot of emax with standard deviation. 
        If groups are given, then on bar per group will be plotted.
        
        :param groups: A dictionary with list patient names.
                       For instance [healthy, HF]. if single = False, 
                       then only the individual patients in each group 
                       with be plotted together. 
        
        """

       
        # Make sure that we have computed the maximum elastance 
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("emax"):
            self.compute("emax")

        logger.info("Plot emax")
            
        outdir = "/".join([self._outdir, "emax2"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = "/".join([outdir, "emax"])

        if groups is None:
            # Plot only one bar
            emax, labels  = [], []
            
            for patient_name in self._keys:
                if self._results[patient_name]["emax"]:
                    emax.append(self._results[patient_name]["emax"])
                    labels.append(patient_name)
                

            tables.print_emax_table(emax, labels)
            plot.plot_emax(emax, labels, path)
        else:

            emax, labels = [], []
            for k, v in groups.iteritems():
                labels.append(k)
                emax_group = []
                for patient_name in v:
                    if patient_name in self._results.keys():
                        emax_group.append(self._results[patient_name]["emax"])
                    else:
                        logger.warning("Patient {} not found".format(patient_name))
                emax.append(emax_group)

            # Do some statistics
            msg = "#"*40 + "\nTesting Emax" + \
                  "Testing if the means of the groups \n{} ".format(labels) + \
                  "are equal using one-way ANOVA:\n"
            logger.info(msg)
            F,p = stats.f_oneway(*emax)
            logger.info("F-value = {}, P-value = {}".format(F, p))  
            logger.info("#"*40)

            # Print a tables
            tables.print_emax_table(emax, labels)
          

            # Plot
            plot.plot_emax(emax, labels, path)

            
    def plot_time_varying_elastance(self):
        """Plot the time varying elastance as a linear curve
        with the time varying elastance being the slope of 
        curve for each time point. These curves are overlayed 
        on the PV loops. 
        
        """

        outdir = "/".join([self._outdir, "time_varying_elastance"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Make sure that we have computed the elastance and volumes
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key("elastance"):
            self.compute("time_varying_elastance")

        if not self._results[keys[0]].has_key("simulated_volume"):
            self.compute("volume")

        logger.info("Plot time varying elastance")

        for patient_name in self._results.keys():

            es = self._results[patient_name]["elastance"]
            v0s = self._results[patient_name]["v0"]
            pressures = self._results[patient_name]["pressure"]
            volumes = self._results[patient_name]["simulated_volume"]
            
            path = "/".join([outdir, patient_name])   
            plot.plot_time_varying_elastance(volumes, pressures, es, v0s, path)
            
    def print_geometric_distance_table(self):
        """Prints out a table with different measures of distance
        between the simulation and the segmentation.
        
        """

        # Make sure that we have computed the geometric distance
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key('geometric_distance'):
            self.compute('geometric_distance')

        logger.info("Print geometric distance table")
        
        labels, mean_dist, max_dist = [], [], []
        
        for patient_name in self._keys:
        
            mean_dist.append(self._results[patient_name]["mean_distance"])
            max_dist.append(self._results[patient_name]["max_distance"])
            labels.append(" ".join(patient_name.split("_")))

        
        tables.print_geometric_distance_table(mean_dist, max_dist, labels, self._label_key)
        tables.print_geometric_distance_table_mean(mean_dist, max_dist)

    def plot_geometric_distance(self, groups = None):
        """Prints out a table with different measures of distance
        between the simulation and the segmentation.
        
        """

        # Make sure that we have computed the geometric distance
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key('geometric_distance'):
            self.compute('geometric_distance')


        outdir = "/".join([self._outdir, "geometric_distance"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        logger.info("Plot geometric distance")

        N = 100
        valve_times = {"passive_filling_begins":0,
                       "mvc": 0,
                       "avo": int(3*N*float(0.05)),
                       "avc":int(3*N*float(0.4)),
                       "mvo":int(3*N*float(0.50))}
        
        labels, mean_dist, labels1 = [], [], []

        
        for patient_name in self._keys:

            d = self._results[patient_name]["mean_distance"]
            d_int = utils.interpolate_trace_to_valve_times(d, self._valve_times[patient_name],N)
            mean_dist.append(d_int)
            labels1.append(" ".join(patient_name.split("_")))
            labels.append(patient_name)
         

        
        path = "/".join([outdir,"mean_geometric_distance_all"])
        plot.plot_geometric_distance(mean_dist, labels1, path, valve_times)

        path = "/".join([outdir,"mean_geometric_distance_groups"])
        plot.plot_geometric_distance(mean_dist, labels, path, valve_times, groups)

        if groups:
            path = "/".join([outdir,"mean_geometric_distance_mean_std"])
            dists = []
            labels = []
            for k, v in groups.iteritems():
                dists.append([])
                labels.append(k)
                for patient_name in v:
                    d = self._results[patient_name]["mean_distance"]
                    d_int = utils.interpolate_trace_to_valve_times(d, self._valve_times[patient_name],N)
                    dists[-1].append(d_int)

            mean_dist = np.mean(dists, 1)
            stds = np.std(dists, 1)
            plot.plot_geometric_distance(mean_dist, labels, path, valve_times, stds=stds)
                    
        

    def print_data_mismatch_table(self):
        """
        Print a table with strain and volume mismatch

        """
        

        # Make sure that we have computed the data mismatch
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key('data_mismatch'):
            self.compute('data_mismatch')

        logger.info("Print data mismatch table")
        
        labels, I_vol, I_strain_rel, I_strain_max = [], [], [], []
        for patient_name in self._keys:

            labels.append(" ".join(patient_name.split("_")))
            I_vol.append(self._results[patient_name]["I_vol"])
            I_strain_rel.append(self._results[patient_name]["I_strain_rel"])
            I_strain_max.append(self._results[patient_name]["I_strain_max"])

        tables.print_data_mismatch_table_mean(I_vol, I_strain_rel, I_strain_max)
        tables.print_data_mismatch_table(I_vol, I_strain_rel, I_strain_max, labels)

    def plot_cardiac_work(self, case = "full", work_pair = "SE"):
        """FIXME! briefly describe function

        :param cases: 
        :param work_pairs: 
        :param region: 
        :returns: 
        :rtype: 

        """
        

        key_str = "{}_{}_region_{}"
        keys = self._results.keys()
        key = key_str.format(work_pair, case, 0)
        if len(keys) == 0 or not self._results[keys[0]].has_key(key):
            try:
                self.compute("cardiac_work:{}:{}".format(case, work_pair))
            except Exception as ex:
                print ex
                
        logger.info("Plot Cardiac Work")

        outdir = "/".join([self._outdir, "cardiac_work"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        
        # Scale the work from mmHg to kPa
        # scale = (101.325/760)
        scale = 1.0
            
        for key in self._keys:
           
            simulated = {}
            measured = {}

            for r in  self._results[key]["measured_work"].keys():

                measured_work = np.multiply(self._results[key]["measured_work"][r], scale)
                
                work = []
                labels = []
                
                assert self._results[key].has_key(key_str.format(work_pair, case, r))
               
                simulated_work = np.array(self._results[key][key_str.format(work_pair, case, r)]["work"])
                              
                work = [simulated_work]

                path = "/".join([outdir, "cardiac_work_{}_{}_region_{}".format(work_pair, case, r)])
                # plot.plot_cardiac_work(work, labels, measured_work, path)

                simulated[r] = simulated_work
                measured[r] = measured_work


        name = "cardiac_work_{}_{}".format(work_pair, case)
        paths = plot.plot_strains({"longitudinal":simulated},
                                  {"longitudinal":measured},
                                  outdir, dirs = ["longitudinal"],
                                  nregions = 12, name = name, scale = False)
                            
                    
    def plot_cardiac_work_scalar(self, workpair = "p_gradu_longitudinal"):
        """FIXME! briefly describe function
        
        :returns: 
        :rtype: 
        
        """

        wp = workpair.split("_")
        assert wp[0] in ["p", "caucy", "piola1", "piola2"]
        assert wp[1] in ["green", "deform", "gradu"]
        assert wp[2] in ["fiber", "longitudinal"]
        
        
        
        # Make sure that we have computed what we need
        keys = self._results.keys()
        if len(keys) == 0 or not self._results[keys[0]].has_key('scalar_features'):
            self.compute("mechanical_features")
            
        if len(keys) == 0 or not self._results[keys[0]].has_key("simulated_strain"):
            self.compute("strain")

        if len(keys) == 0 or not self._results[keys[0]].has_key("meausured_work"):
            self.compute("measurements")

            
        logger.info("Plot Cardiac Work")
       
        outdir = "/".join([self._outdir, "cardiac_work_scalar"])
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Scale the work from mmHg to kPa
        scale = (101.325/760)

        simulated = {}
        measured = {}
            
        for key in self._keys:

            
            for r in self._results[key]["measured_work"].keys():

                measured_work = np.multiply(self._results[key]["measured_work"][r], scale)

                

                if workpair == "p_gradu_longitudinal":
                    strain = self._results[key]["simulated_strain"]["longitudinal"][r]
                    pressure = self._results[key]["pressure"][:len(strain)]
                    simulated_work = utils.compute_cardiac_work_echo(pressure, strain, flip=True)

                # elif workpair == "p_gradu_fiber":
                #     strain = self._results[key]["simulated_strain"]["longitudinal"][r]
                #     pressure = self._results[key]["pressure"][:len(strain)]
                #     simulated_work = utils.compute_cardiac_work_echo(pressure, strain, flip=True)

                else:
                    stress_keystr = "{}_{}_stress".format(wp[0], wp[2])
                    strain_keystr = "{}_{}_strain".format(wp[1], wp[2])

                    strain = self._results[key]["features_scalar"][strain_keystr][str(r)]
                    stress = self._results[key]["features_scalar"][stress_keystr][str(r)]
          
                    simulated_work = utils.compute_cardiac_work_echo(stress, strain, flip=False)

                    
                work = [simulated_work]
                labels = ["simulated"]

                path = "/".join([outdir, "cardiac_work_region_{}".format(r)])
                # plot.plot_cardiac_work(work, labels, measured_work, path)

                simulated[r] = simulated_work
                measured[r] = measured_work
                
            paths = plot.plot_strains({"longitudinal":simulated},
                                      {"longitudinal":measured},
                                      outdir, dirs = ["longitudinal"],
                                      nregions = 12, name = workpair, scale = False)

        
        
        
    def make_simulation(self, refined=False, level = 2):

        setup_general_parameters()

        # Make sure that we have computed the features
        keys = self._features.keys()
        if len(keys) == 0:
            self.compute("vtk_simulation")
            
        logger.info("#"*40+"\nMake simulation")
            
        main_outdir = "/".join([self._outdir, "simulation"])
        if not os.path.exists(main_outdir):
            os.makedirs(main_outdir)


        
        
        for patient_name in self._results.keys():

            logger.info("\tPatient: {}".format(patient_name))
            outdir = "/".join([main_outdir, patient_name])
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            params, patient  = self._setup_compute(patient_name)
            # params = self._params
                
            # params["Patient_parameters"]["patient"] = patient_name
            # params["Patient_parameters"]["mesh_path"] \
            #     = self._mesh_path.format(patient_name)
            # params["Patient_parameters"]["pressure_path"] \
            #     = self._pressure_path.format(patient_name)

            
            
            # utils.save_displacements(params, self._features[patient_name], outdir)
            if refined:
                utils.make_refined_simulation(params,
                                              self._features[patient_name],
                                              outdir,
                                              patient,
                                              self._data[patient_name])
            else:
                utils.make_simulation(params, self._features[patient_name], outdir,
                                      patient, self._data[patient_name])
            
        logger.info("#"*40)

    def snap_shots(self, feature = "", feature_space = "DG_0"):

        setup_general_parameters()

        # Make sure that we have computed the features
        keys = self._features.keys()
        if len(keys) == 0:
            self.compute("vtk_simulation")

        if feature != "":
            all_features = self._features[self._keys[0]].keys()
            msg = "Unknown feature {}, possible inputs are: \n{}".format(feature, all_features)
            assert feature in all_features, msg

        main_outdir = "/".join([self._outdir, "snap_shots"])
        if not os.path.exists(main_outdir):
            os.makedirs(main_outdir)

            
        params = self._params
        
        for patient_name in self._keys:

            fs = self._features[patient_name][feature]
            us = self._features[patient_name]["displacement"]


            params["Patient_parameters"]["patient"] = patient_name
            params["Patient_parameters"]["mesh_path"] \
                = self._mesh_path.format(patient_name)
            params["Patient_parameters"]["pressure_path"] \
                = self._pressure_path.format(patient_name)
            
            
            outdir = "/".join([main_outdir, patient_name, feature])
            vtk_utils.make_snapshots(fs, us, feature_space, outdir, params)
            
    def set_feature_keys(self, *args):
        self._feature_keys = args
