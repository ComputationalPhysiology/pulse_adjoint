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
# import os
# import logging

# log_level = logging.INFO

# # OPTIMIZATION PARAMETERS #

# # Strain weights for optimization
# # What kind of weighting rule
# WEIGHT_RULES = ["equal", "drift", "peak_value", "combination"]
# DEFAULT_WEIGHT_RULE = "equal"
# # Any preferred direction for weighting ("c", "l", "r" or None)
# WEIGHT_DIRECTIONS = ["c", "l", "r", "all"]
# DEFAULT_WEIGHT_DIRECTION = "all"

# # The different phases we can optimize
# PHASES = ['passive_inflation', 'active_contraction', "all"]

# # If true, Optimize material parameters, otherswise use
# # the default material parameters
# OPTIMIZE_MATPARAMS = True

# GAMMA_INC_LIMIT = 0.02

# # Max size of gamma
# MAX_GAMMA = 0.9


# # Optimization method
# # DEFAULT_OPTIMIZATION_METHOD = "SLSQP"
# OPTIMIZATION_METHODS = ["TNC", "L-BFGS-B", "SLSQP", "ipopt"]


# # SYNTHETIC DATA #
# NSYNTH_POINTS = 7
# SYNTH_PASSIVE_FILLING = 3

# # LABELS AND NAMES #

# STRAIN_REGIONS = {"LVBasalAnterior": 1,
#                   "LVBasalAnteroseptal": 2,
#                   "LVBasalSeptum": 3,
#                   "LVBasalInferior": 4,
#                   "LVBasalPosterior": 5,
#                   "LVBasalLateral": 6,
#                   "LVMidAnterior": 7,
#                   "LVMidAnteroseptal": 8,
#                   "LVMidSeptum": 9,
#                   "LVMidInferior": 10,
#                   "LVMidPosterior": 11,
#                   "LVMidLateral": 12,
#                   "LVApicalAnterior": 13,
#                   "LVApicalSeptum": 14,
#                   "LVApicalInferior": 15,
#                   "LVApicalLateral": 16,
#                   "LVApex": 17}

# STRAIN_DIRECTIONS = ["RadialStrain", "LongitudinalStrain",
#                      "CircumferentialStrain", "AreaStrain"]
# # Strain regions
# STRAIN_REGION_NUMS = STRAIN_REGIONS.values()
# STRAIN_REGION_NUMS.sort()
# STRAIN_NUM_TO_KEY = {0: "circumferential",
#                      1: "radial",
#                      2: "longitudinal"}

# PASSIVE_INFLATION_GROUP = "passive_inflation"
# CONTRACTION_POINT = "contract_point_{}"
# ACTIVE_CONTRACTION = "active_contraction"
# ACTIVE_CONTRACTION_GROUP = "/".join([ACTIVE_CONTRACTION, CONTRACTION_POINT])
# # PASSIVE_INFLATION = "passive_inflation"

# # DIRECTORIES AND PATHS #

# # Folders and path for which the data is stored in .h5 format
# curdir = os.path.abspath(os.path.dirname(__file__))
# DEFAULT_SIMULATION_FILE = os.path.join(curdir, 'local_results/results.h5')

# # DOLFIN PARAMETERS #
# # Nonlinear solver
# NONLINSOLVER = "snes"

# # Nonlinear method
# # (Dolfin Adjoint version < 1.6 newtontr/ls are the only one working)
# SNES_SOLVER_METHOD = "newtontr"

# # Maximum number of iterations
# SNES_SOLVER_MAXITR = 15

# # Absolute Tolerance
# SNES_SOLVER_ABSTOL = 1.0e-5

# # Linear solver "
# SNES_SOLVER_LINSOLVER = "lu"
# # SNES_SOLVER_LINSOLVER = "mumps"
# SNES_SOLVER_PRECONDITIONER = "default"

# # Print Non linear solver output
# VIEW_NLS_CONVERGENCE = False
