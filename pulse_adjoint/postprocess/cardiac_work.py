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
from .args import *


class CardiacWork(object):
    r"""
    Cardiac Work can be copmuted in several ways using 
    conjuate stress-strain pairs:

    +-----------------------------------+-------------------------------------------+
    | Stress                            | Strain                                    |
    +===================================+===========================================+
    | Second Piola (:math:`\mathbf{S}`) | Green Lagrange (:math:`\mathbf{E}`)       |
    +-----------------------------------+-------------------------------------------+
    | First Piola (:math:`\mathbf{P}`)  | Deformation gradient (:math:`\mathbf{F}`) |
    +-----------------------------------+-------------------------------------------+

    For this example we consider the Second Piola :math:`\mathbf{S}`
    and the Green Lagrange :math:`\mathbf{E}`.

    Since we cosider a quasi-static model, the work 
    is computed as the average between two timepoints.
    
    The work is computed in the following steps:


    1. Compute the average stress for time :math:`t_i` and
       :math:`t_{i+1}`:

    .. math:: 
      
       \mathbf{S} &= \frac{\partial \Psi}{\partial \mathbf{E}},   \\    
       \overline{\mathbf{S}}_{t_i} &= \frac{1}{2} \left( \mathbf{S}_{t_i}+ \mathbf{S}_{t_{i+1}} \right)

    where :math:`\Psi` is the strain energy density function. 

    2. Compute the strain rate

    .. math:: 
      
       \mathbf{E} &= \frac{1}{2}\left(\mathbf{C} - \mathbf{I} \right) \\      
       \mathrm{d}\mathbf{E}_{t_i} &= - \left( \mathbf{E}_{t_{i+1}} - \mathbf{E}_{t_{i+1}} \right)

    where :math:`\mathbf{C}` is the right Cauchy Greeen strain tensor and 
    :math:`\mathbf{I}` is the identity.
    I you want to compute the work done in a certain direction, 
    i.e along the fibers, :math:`\mathbf{e}_{f}`, then use

    .. math::
    
       \mathbf{S}_{f} &= \mathbf{e}_{f} \cdot (\mathbf{S} \mathbf{e}_{f}) \\
       \mathbf{E}_{f} &= \mathbf{e}_{f} \cdot (\mathbf{E} \mathbf{e}_{f})

    3. Compute the power:

    .. math::

       \mathcal{P}_{t_i} = \frac{1}{|\Omega|}\int_{\Omega} \overline{\mathbf{S}}_{t_i} : 
       \mathrm{d}\mathbf{E}_{t_i} \mathrm{d}V

    4. Compute the work:

    .. math::

       W_n = \sum_{i}^{n} \mathcal{P}_{t_i}


    Note that the power is really copmuted using the strain rate which is given by

    .. math::
    
       \frac{\partial \mathbf{E}}{\partial t} = \frac{ \mathbf{E}_{t_{i+1}} - \mathbf{E}_{t_{i+1}}}{t_{i+1} - t_i}, 

    but the time increment cancels in the integration of the power, when computing the work.
    Therefore we do not include this in the implementation. 

    .. note::
    
       Stress, :math:`\overline{\mathbf{S}}_{t_i}` has units kPa, while 
       :math:`\mathrm{d}\mathbf{E}_{t_i}` is unitless. Since we divide 
       by the volume of the region of interest, when computing the power 
       we end up with a work per unit volum, i.e unit Joule/:math:`m^3`.    
    
    """
    
    def __init__(self, V, W):
        """
        Intitialize CardiacWork class

        :param V: Space where to project the stresses and strains
        :type V: :py:class:`dolfin.FunctionSpace`

        """

        self._V = V
        self._W = W
        
        self.reset()
        self._print_head()
        


    def __call__(self, strain_tensor, stress_tensor, case, e_k):
        """FIXME! briefly describe function

        :param strain_tensor: 
        :type strain_tensor: :py:class:`dolfin.Function`
        :param stress_tensor: 
        :type stress: :py:class:`dolfin.Function`
        :returns: 
        :rtype: 

        """
        
        assert case in ["full", "comp"], "Unknown case {}".format(case)
        
        self._strain_tensor = strain_tensor
        self._stress_tensor = stress_tensor
        
        # Strain rate (ish)
        self._dstrain = self._strain_tensor-self._strain_tensor_prev
        # Average stress
        self._stress_avg = 0.5*(self._stress_tensor + self._stress_tensor_prev)

        # Compute cardiac work
        self.compute_cardiac_work(case, e_k)
        

    def reset(self):

        self._strain_tensor_prev = dolfin.Function(self._V, name = "strain_tensor_prev")
        self._stress_tensor_prev = dolfin.Function(self._V, name = "stress_tensor_prev")
        self._work = []
        self._power = []
        
    def compute_cardiac_work(self, case, e_k = None):
        """
        Compute Cardac work, and store the values in a list.
        The results can be access through self.get_results()

        :param str case: A string saying if you want to compute the
                         work along a direction (specified by e_k), or 
                         if you want to compute the total work.
                         Possible inputs are ('full', 'comp')
        :param dx: A volume measure used in the integration
        :type dx: :py:class:`dolfin.Measure`
        :param e_k: A vectorfield, for which you want to compute the work along.
        :type e_k: :py:class:`dolfin.Function`

        """
        
        
        
        
        S = self._get_stress(case, e_k)
        dE = self._get_strain_rate(case, e_k)

        # Compute power
        P = self._compute_power(S, dE)

        self._power.append(P.copy())
        
        # The work is just the cumulative sum
        from ..utils import list_sum
        self._work.append(list_sum(self._power))

    
        # self._print_line()
        self._assign_prev()

    def get_power(self):
        return self._power[-1]
    def get_work(self):
        return self._work[-1]
    
    def _assign_prev(self):

        S = dolfin.project(self._stress_tensor, self._V)
        E = dolfin.project(self._strain_tensor, self._V)
        self._strain_tensor_prev.assign(E)
        self._stress_tensor_prev.assign(S)

    def _compute_power(self, S, dE):
        
        return dolfin.project(dolfin.inner(S, dE), self._W)

    def _print_head(self):

        print("\n\t{:<10}\t{:<10}\t{:<10}".format("Region", "Power", "Work"))
        
    def _print_line(self):

        print("\t{:<10.2f}\t{:<10.2f}".format(self._power[-1],
                                              self._work[-1]))


    def _get_stress(self, case, e_k = None):
        
        if case == "full":
            S = self._stress_avg

        elif case == "comp":
            msg = "Please provide a vectorfield to the contructor"
            assert e_k is not None, msg
            S = dolfin.inner(self._stress_avg*e_k, e_k)
            

        return S
            
    
    def _get_strain_rate(self, case, e_k = None):

        if case == "full":
            dE = self._dstrain

        elif case == "comp":
            msg = "Please provide a vectorfield to the contructor"
            assert e_k is not None, msg
            dE = dolfin.inner(self._dstrain*e_k, e_k)

        return dE
            
    def get_results(self):
        
        return {"power": self._power, "work": self._work}
        
            
            
    
class CardiacWorkEcho(CardiacWork):
    r"""
    This is a class for computing cardiac work, when
    the stress is substituted by a scalar pressure. 
    This is similar to what is done in EchoPac

    According to [1] one start with the left
    veintricular cavity pressure (:math:`p_{\mathrm{lv}}`) and the average strain
    (:math:`\varepsilon`) (could be segemental). Then one differentiates the strain
    in order to obtain the strain-rate (:math:`\dot{\varepsilon}`). Then one
    multiply the strain-rate with the pressure to obain the power
    :math:`\mathcal{P}`, and finally one integrate the power over time to obtain
    the work (:math:`W`). Written out:

    .. math::
      
       \dot{\varepsilon}_i &=  \frac{ \varepsilon_{i+1} - 
       \varepsilon_{i}}{t_{i+1} - t_{i}}, \;\; i = 1 \cdots N - 1 \\
       \bar{p}_i &= - \frac{ p_{i+1} + p_{i}}{2},  \;\; i = 1 \cdots N - 1\\
       \mathcal{P}(t_i) &= \dot{\varepsilon}_i \cdot \bar{p}_i =  
       -  \frac{ \varepsilon_{i+1} - \varepsilon_{i}}{t_{i+1} - t_{i}} 
       \cdot \frac{ p_{i+1} + p_{i}}{2},  \;\; i = 1 \cdots N - 1 \\
       W(t_i) &= \int_0^{t_i} \mathcal{P}(t^*)dt^* \\
       & \approx \sum_{j = 1}^{i} \mathcal{P}(t_j) \cdot (t_{j+1} - t_{j}) \\
       &=  \sum_{j = 1}^{i}  \dot{\varepsilon}_j \cdot \bar{p}_j   
       \cdot (t_{j+1} - t_{j}) \\
        &=  \sum_{j = 1}^{i}  - \frac{ \varepsilon_{j+1} -
        \varepsilon_{j}}{t_{j+1} - t_{j}} \cdot \frac{ p_{j+1} + p_{j}}{2} 
        \cdot (t_{j+1} - t_{j}) \\
        &= -\sum_{j = 1}^{i}  \frac{ (\varepsilon_{j+1} -
        \varepsilon_{j})(p_{j+1} + p_{j})}{2} \\


    .. rubric:: Reference

    [1] Russell, Kristoffer, et al. "Assessment of wasted myocardial work: 
    a novel method to quantify energy loss due to uncoordinated left ventricular 
    contractions." American Journal of Physiology-Heart and Circulatory 
    Physiology 305.7 (2013): H996-H1003.
    """
    
    def reset(self):
        
        self._strain_tensor_prev = dolfin.Function(self._V, name = "strain_tensor_prev")
        self._stress_tensor_prev = 0.0
        self._work = []
        self._power = []

    def _get_stress(self, case, *args):

        if case == "full":
            return -self._stress_avg*dolfin.Identity(self._V.mesh().geometry().dim())

        else:
            return -self._stress_avg

    def _get_strain_rate(self, case, e_k = None):

        if case == "full":
            return self._dstrain
            # raise ValueError("Cannot compute total work for echo work")

        elif case == "comp":
            msg = "Please provide a vectorfield to the contructor"
            assert e_k is not None, msg
            dE = dolfin.inner(self._dstrain*e_k, e_k)

        return dE

    def _assign_prev(self):

        E = dolfin.project(self._strain_tensor, self._V)
        self._strain_tensor_prev.assign(E)
        self._stress_tensor_prev = self._stress_tensor
              

              

class StrainEnergy(object):
    def __init__(self):
        
        self._print_head()

    def __call__(self, psi, dx):
        
        meshvol = dolfin.assemble(dolfin.Constant(1.0)*dx)
        psi_avg = dolfin.assemble(psi*dx)/meshvol
        self._strain_energy.append(psi_avg)
        self._print_line()
    def _print_head(self):

        print("\n\t{:<10}".format("Work"))
        

    def _print_line(self):
        print("\n\t{:<10}".format(self.strain_energy[-1]))
        
    def reset(self):
        self._strain_energy = []
        
    def get_results(self):
        return {"work": self._strain_energy}


def work_trace(pressure, strain):

    import numpy as np
    assert len(pressure) == len(strain)

    pressure_avg = np.add(pressure[:-1], pressure[1:])/2.0
    dstrain = -np.diff(s)
    work = np.cumsum(dstrain*pressure_avg)
    
