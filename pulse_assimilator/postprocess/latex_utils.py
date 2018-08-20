#!/usr/bin/env python
"""
This script includes functionality for putting figures together
by putting the figures into a latex document, compiling it, 
and move the pdf to the desired location
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
import os, shutil
import numpy as np
from .tables import tabalize


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
latex_head = r"""\documentclass[tightpage, 26pt]{{standalone}}
\usepackage{{subcaption}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage[export]{{adjustbox}}
\newcolumntype{{C}}{{>{{\centering\arraybackslash}} m{{0.12\textwidth}}}}  

\newcommand{{\imgcasefront}}[1]{{\adjincludegraphics[scale=0.04,trim={{{{.2\width}} {{.01\height}} {{.2\width}} {{.01\height}}}}, clip]{{{0}_#1_front}}}}
\newcommand{{\imgcaseside}}[1]{{\adjincludegraphics[scale=0.04,trim={{{{.2\width}} {{.01\height}} {{.2\width}} {{.01\height}}}}, clip]{{{0}_#1_side}}}}

\begin{{document}}
\setlength\tabcolsep{{0.0pt}}
\renewcommand{{\arraystretch}}{{0.0}}
"""

def tab_head(n):
    s = r"""
    \begin{tabular}{"""+r"""C"""*n+r"""}
    """
    return s.replace(" ", "")

def tab_labels(n):
    s = r"""
    """.replace(" ", "")
    for i in range(n):
        s+=r"""\multicolumn{{{{1}}}}{{{{l}}}}{{{{{{{}}}}}}} \vspace{{{{0.01cm}}}}""".format(i)
        if i == n-1:
            s+= r"""\\
"""
        else:
            s+= r"""&
"""
    return s

def tab_img(n):
    s = r"""
    """.replace(" ", "")
    for i in range(n):
        s+=r"""\imgcasefront{{{{{{{}}}}}}} \vspace{{{{0.01cm}}}}""".format(i)
        if i == n-1:
            s+= r"""\\
"""
        else:
            s+= r"""&
"""
    for i in range(n):
        s+=r"""\imgcaseside{{{{{{{}}}}}}} \vspace{{{{0.01cm}}}}""".format(i)
        if i == n-1:
            s+= r"""\\
"""
        else:
            s+= r"""&
"""
    return s

def tab_heatmap(n):
     return r"""
\multicolumn{{{{{0}}}}}{{{{c}}}}{{{{{{1}}}}}} \vspace{{{{0.1cm}}}}\\
\multicolumn{{{{{0}}}}}{{{{c}}}}{{{{\adjincludegraphics[scale=0.1,trim={{{{{{{{.0\width}}}} {{{{.0\height}}}} {{{{.0\width}}}} {{{{.785\height}}}}}}}}, clip]{{{{{{0}}}}}}}}}}\\
""".format(n)
    

tab_tail=r"""
\end{tabular}
\end{document}
"""

strain_tab=r"""\documentclass[tightpage, 26pt]{{standalone}}
\usepackage{{subcaption}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{array}}
\usepackage[export]{{adjustbox}}
\setlength{{\tabcolsep}}{{0pt}}%
\renewcommand{{\arraystretch}}{{0}}
\begin{{document}}
\begin{{tabular}}{{c}}
 \includegraphics[scale =0.25]{{{0}}} \\
\hline
  \includegraphics[scale = 0.25]{{{1}}} \\
\hline
  \includegraphics[scale = 0.25]{{{2}}} \\
\end{{tabular}}
\end{{document}}
"""



def make_canvas_strain(paths, name = None):

    outdir = os.path.dirname(paths[0])
   
    latex_full = strain_tab.format(*paths)

    fname = "simulated_strains" if name is None else name
    
    fnametex = ".".join([fname, "tex"])
    with open(fnametex, "w") as f:
        f.write(latex_full)

    os.system("pdflatex {} >/dev/null".format(fnametex))
   
    for ext in [".aux", ".log", ".tex"]:
        os.remove(fname+ext)

    src= ".".join([fname, "pdf"])
    dst = "/".join([outdir, src])
  
    shutil.move(src, dst)
    print "moved from {} to {}".format(src, dst)
    

def make_canvas_snap_shot(lst, times, name, heatmap_name = "", heatmap_label = r"$\gamma$"):

    assert len(lst) == len(times), \
        "Not equal length"

    div = False
    for n in [4,5,6]:
        if not(len(lst) % n):
            N = n
            div = True

    assert div, \
        "list must be diviable by four six or five. Length is {}".format(len(lst))

    # Check that the files exist and add extension
    for l in lst:
        v = "_{}".format(l)
        for s in ["_front", "_side"]:
            if os.path.isfile(name + v + s):
                shutil.move(name + v + s,
                            name + v + s +".png")
            else:
                if not os.path.isfile(name + v + s + ".png"):
                    raise IOError("File {} not found".format(name + v + s))
                
    if heatmap_name != "":
        if not os.path.isfile(heatmap_name + ".png"):
            if os.path.isfile(heatmap_name):
                shutil.move(heatmap_name, heatmap_name+".png")
                latex_heatmap = tab_heatmap(N).format(heatmap_name,
                                                      heatmap_label)
            else:
                print "Not heatmap figure named ", heatmap_name
                print "Make figure without heatmap"
                latex_heatmap = ""

        else:
            latex_heatmap = tab_heatmap(N).format(heatmap_name,
                                                  heatmap_label)
    
       
    else:
        latex_heatmap = ""
    
    
    
            
    lst_chunks = [lst[i:i+N] for i in range(0, len(lst), N)]
    times_chunks = [times[i:i+N] for i in range(0, len(times), N)]


    latex_full = latex_head.format(name)+\
                 tab_head(N) + latex_heatmap

    for t,l in zip(times_chunks,lst_chunks):
        latex_full += tab_labels(N).format(*t)
        latex_full += tab_img(N).format(*l)

    latex_full += tab_tail

    fname = "_".join(["snap_shots", os.path.basename(name)])
    fnametex = ".".join([fname, "tex"])
    with open(fnametex, "w") as f:
        f.write(latex_full)

    os.system("pdflatex {} >/dev/null".format(fnametex))
   
    for ext in [".aux", ".log", ".tex"]:
        os.remove(fname+ext)

    src= ".".join([fname, "pdf"])
    dst = "/".join([os.path.abspath(os.path.dirname(name)), src])
  
    shutil.move(src, dst)
    print "moved from {} to {}".format(src, dst)
    


if __name__ == "__main__":


    lst = range(3,26, 2)
    lst2 = [ r"{:.0f} $\%$".format(i) for i in np.linspace(0,100, len(lst))]
    for i in lst2: print i 
  
    
    make_canvas_snap_shot(lst, lst2, "gamma", "heatmap.png")



