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
import dolfin
import numpy as np
from ..setup_optimization import merge_control
from ..config import *

ALL_ACTIVE_GROUP = "alpha_{}/reg_par_{}/active_contraction/contract_point_{}"
ALL_PASSIVE_GROUP = "alpha_{}/reg_par_0.0/passive_inflation"

STRAIN_REGION_NAMES = {1:"Anterior",
                       2:"Anteroseptal",
                       3:"Septum",
                       4:"Inferior",
                       5:"Posterior",
                       6:"Lateral",
                       7:"Anterior",
                       8:"Anteroseptal",
                       9:"Septum",
                       10:"Inferior",
                       11:"Posterior",
                       12:"Lateral",
                       13:"Anterior",
                       14:"Septum",
                       15:"Inferior",
                       16:"Lateral",
                       17:"Apex"}

STRAIN_REGIONS = {1:"LVBasalAnterior",
                 2:"LVBasalAnteroseptal",
                 3:"LVBasalSeptum",
                 4:"LVBasalInferior",
                 5:"LVBasalPosterior",
                 6:"LVBasalLateral",
                 7:"LVMidAnterior",
                 8:"LVMidAnteroseptal",
                 9:"LVMidSeptum",
                 10:"LVMidInferior",
                 11:"LVMidPosterior",
                 12:"LVMidLateral",
                 13:"LVApicalAnterior",
                 14:"LVApicalSeptum",
                 15:"LVApicalInferior",
                 16:"LVApicalLateral",
                 17:"LVApex"}

try:
    dolfin.parameters["allow_extrapolation"] = True
except:
    pass
work_pairs = ["SE", "PF", "pgradu", "strain_energy", "SEdev"]
cases = ["full", "comp_fiber", "comp_long"]
