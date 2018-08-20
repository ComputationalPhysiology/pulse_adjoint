#!/usr/bin/env python
"""
This script includes functionality to put data in table format,
which can just be pasted directly into you latex document.
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
try:
    import tabulate
    tabulate.LATEX_ESCAPE_RULES = {}
except:
    has_tabulate=False
else:
    has_tabulate=True

def print_error():
    print "Warning tabulate module not found"
    print "To print tables please install tabulate"
    print "pip install tabulate"
    raise ImportError

import numpy as np

def tabalize(caption, header, table, label, floatfmt=".2e"):
    if not has_tabulate: print_error()

    tabular =  tabulate.tabulate(table, header,
                                 tablefmt="latex", floatfmt=floatfmt)
    T = \
        r"""
\begin{{table}}
\caption{{{}}}
{}
\label{{{}}}
\end{{table}}
""".format(*[caption, tabular, label])
        
    return T

def print_geometric_distance_table_mean(mean_dist, max_dist):

    lst = np.array([[np.mean([np.mean(m) for m in mean_dist])],
                    [np.max([np.mean(m) for m in mean_dist])],
                    [np.max([np.max(m) for m in mean_dist])],
                    [np.mean([np.mean(m) for m in max_dist])],
                    [np.max([np.mean(m) for m in max_dist])],
                    [np.max([np.max(m) for m in max_dist])]]).T

    caption = "Distance between simualtion and segmenetation"
    header = [r"$\langle \langle \overline{d}(\Xi^{q,i}) \rangle_i \rangle_q$",
              r"$\max_q \langle \overline{d}(\Xi^{q,i})\rangle_i$",
              r"$\max_q \max_i \overline{d}(\Xi^{q,i})$",
              r"$\langle \langle d_{\max}(\Xi^{q,i}) \rangle_i \rangle_q$",
              r"$\max_q \langle d_{\max}(\Xi^{q,i})\rangle_i$",
              r"$\max_q \max_i d_{\max}(\Xi^{q,i})$"]
    table = lst
    label = "tab:seg_comp"
        
    T = tabalize(caption, header, table, label, floatfmt=".3g")
    print(T)

def print_geometric_distance_table(mean_dist, max_dist, labels, label_key = ""):

    lst = np.array([labels,
                   [np.mean(m) for m in mean_dist],
                   [np.max(m) for m in mean_dist],
                   [np.mean(m) for m in max_dist],
                    [np.max(m) for m in max_dist]]).T

    caption = "Distance between simualtion and segmenetation"
    header = [label_key,
              r"$\langle \overline{d}(\Xi^{q,i}) \rangle_i$",
              r"$\max_i \overline{d}(\Xi^{q,i})$",
              r"$\langle d_{\max}(\Xi^{q,i}) \rangle_i$",
              r"$\max_i d_{\max}(\Xi^{q,i})$"]
    table = lst
    label = "tab:seg_comp"
        
    T = tabalize(caption, header, table, label, floatfmt=".3g")
    print(T)

def print_data_mismatch_table_mean(I_vol, I_strain_rel, I_strain_max):
    
    # Print mean and stds
    lst = np.array([["{:.3g} $\pm$ {:.2g}".format(np.mean(I_vol), np.std(I_vol))],
                    ["{:.3g} $\pm$ {:.2g}".format(np.mean(I_strain_rel), np.std(I_strain_rel))],
                    ["{:.3g} $\pm$ {:.2g}".format(np.mean(I_strain_max), np.std(I_strain_max))]]).T
    print [np.std(I_vol), np.std(I_strain_rel), np.std(I_strain_max)]

    caption = "Data mismatch"
    header = [r"$\Ivolavg$",
              r"$\Istrainavg$",
              r"$\Istrainrelmax$"]
    table = lst
    label = "tab:data_mismatch"
    
    T = tabalize(caption, header, table, label, floatfmt=".3g")
    print(T)

def print_data_mismatch_table(I_vol, I_strain_rel, I_strain_max, labels):
    
    # Print mean and stds
    lst = np.array([labels, I_vol, I_strain_rel, I_strain_max]).T

    caption = "Data mismatch"
    header = [r"$\theta$",
              r"$\Ivolavg$",
              r"$\Istrainavg$",
              r"$\Istrainrelmax$"]
    table = lst
    label = "tab:data_mismatch"
    
    T = tabalize(caption, header, table, label, floatfmt=".3g")
    print(T)

def print_emax_table(emax, labels):

    if isinstance(emax[0], (list, tuple, np.ndarray)):
        
        lst = np.array([["{:.3g} $\pm$ {:.2g}".format(np.mean(e), np.std(e))] \
                        for e in emax]).T
        
    else:
       
        lst = np.array([emax])

    caption = "Emax"
    header =  labels
    table = lst
    label = "tab:data_mismatch"
        
    T = tabalize(caption, header, table, label, floatfmt=".3g")
    print(T)
