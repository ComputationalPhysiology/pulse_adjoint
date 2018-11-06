#!/usr/bin/env python
"""
This script includes functionality for plotting the results.
There are different plotting functionalities to plot different
results. 

There is also a setup function where you can speficy options for the plotting.

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
import matplotlib as mpl
from matplotlib import pyplot as plt, rcParams, cbook, ticker, cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import warnings, os

import numpy as np
from scipy import stats


try:
    import seaborn as sns
except:
    has_seaborn = False
else:
    has_seaborn = True

    
# Color map
cmap = plt.get_cmap('gist_rainbow')

def get_colormap(ncolors, transparent = False):

    if transparent:
        
        if ncolors == 1:
            return "gray"
        elif ncolors == 2:
            return [sns.xkcd_rgb["pale red"],
                    sns.xkcd_rgb["denim blue"]]
            # return [sns.xkcd_rgb["light blue"],
            #     sns.xkcd_rgb["pale pink"]]
        elif ncolors == 3:
            return [sns.xkcd_rgb["pale red"],
                    sns.xkcd_rgb["denim blue"],
                    sns.xkcd_rgb["medium green"]]

    if ncolors == 1:
        return "k"
    elif ncolors == 2:
        return [sns.xkcd_rgb["pale red"],
                sns.xkcd_rgb["denim blue"]]
    elif ncolors == 3:
        return [sns.xkcd_rgb["pale red"],
                sns.xkcd_rgb["denim blue"],
                sns.xkcd_rgb["medium green"]]
    elif ncolors == 4:
        return [sns.xkcd_rgb["pale red"],
                sns.xkcd_rgb["denim blue"],
                sns.xkcd_rgb["medium green"],
                sns.xkcd_rgb["amber"]]
    else:
        return sns.color_palette("Paired", ncolors)
# cmap = plt.get_cmap('jet')
# cmap = plt.cm.get_cmap('RdYlBu')

# Linestyles
lineStyles = ['-', '--', ':']
markers = ['o','+'] 

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

    
def setup_plot(fontsize=7,dpi=300,height=9.5, width=9.5, autolayout=True, save_format="png"):
    
    # Plotting options
    if has_seaborn:
        sns.set_palette("husl")
        sns.set_style("white")
        sns.set_style("ticks")
    mpl.rcParams.update({'figure.autolayout': autolayout})

  
    
    mpl.rcParams['font.family']= 'times'
    mpl.rcParams['font.weight']= 'normal'
    mpl.rcParams['font.size']= fontsize
    mpl.rcParams['font.size']= fontsize

    # width in cm

    # minimal size
    # width = 3
    # # single column
    # width = 9
    # # 1.5 column
    # width = 14
    # # full width
    # width = 19

    # # For two images side by side, leave some space
    # width = 9.5
    # height = 9.5
    
    # width = 19.5
    # height = 19.5
    inch = cm2inch(width, height)
    print inch
    
    mpl.rcParams['figure.figsize']= inch #(2.5,2.5)
    
    

    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.titlesize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize

    # mpl.rcParams['figure.dpi'] = 30
    mpl.rcParams['savefig.dpi'] = dpi
    mpl.rcParams['savefig.format'] = save_format
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode']=True
    
    # Surpress warnings from matplotlib
    warnings.filterwarnings("ignore", module="matplotlib")

def plot_gamma_mean_std(means, stds, path, labels = None, valve_times = None):

    setup_plot()

    colors = get_colormap(len(means))
    colors_trans = get_colormap(len(means), True)
    labels = [""]*len(means) if labels is None else labels
    n = len(means[0])
    x = range(n)
    
    fig = plt.figure()
    # fig.set_rasterized(True)
    ax = fig.gca()

    ax.set_xlabel("Valvular event")
    ax.set_ylabel(r"$\overline{\gamma}$")
    

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.fill_between(x, m + s, m - s,  facecolor = colors_trans[i])#, alpha = 0.5)
        ax.plot(x,m, label = labels[i], color = colors[i])
        

    if valve_times is not None:

        echo_valve_times = valve_times
        pfb = valve_times.pop("passive_filling_begins")
        # vs = ["mvc", "avo", "avc", "mvo"]
        # [echo_valve_times[v] for v in vs]
        vals = [(a-pfb)%n for a in echo_valve_times.values()]
        keys = [k for k in echo_valve_times.keys()]
        ax.set_xticks(vals)
        ax.set_xticklabels(keys, rotation = 45)

    ax.legend(loc = "best")
    # ax.set_rasterized(True)

    fig.savefig(path)#, rasterized = True)
    
    
def plot_single_mean_gamma(gamma, path, valve_times = None):

    if len(gamma) == 0: return
    setup_plot()
    
    fig = plt.figure()
    ax = fig.gca()

    n =  len(gamma)
    x = range(n)
    ax.plot(x, gamma, "k-o")
    
    ax.set_xlabel("Point")
    ax.set_ylabel(r"$\overline{\gamma}$")
    if valve_times is not None:
        echo_valve_times = valve_times["echo_valve_time"]
        pfb = valve_times["passive_filling_begins"]
        vals = [(a-pfb)%n for a in echo_valve_times.values()]
        keys = [k for k in echo_valve_times.keys()]
        ax.set_xticks(vals)
        ax.set_xticklabels(keys)
        
    fig.savefig(path, format="png")


def plot_multiple_gamma(gammas, path, labels = None,
                        valve_times = None):

    if len(gammas) == 0: return
    setup_plot()
    
    m = len(gammas)
    colors = [cmap(i) for i in np.linspace(0, 1, m)]
    linestyles = lineStyles*(int(m/3.)+1)
    fig = plt.figure()
    ax = fig.gca()

    
    for i, g in enumerate(gammas):
        n =  len(g)
        x = range(n)
        
        ax.plot(x, g, label = labels[i],
                color = colors[i], linestyle = linestyles[i])
        
    ax.set_xlabel("Point")
    ax.set_ylabel(r"$\overline{\gamma}$")
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    
    if valve_times is not None:
        echo_valve_times = valve_times["echo_valve_time"]
        pfb = valve_times["passive_filling_begins"]
        vals = [(a-pfb)%n for a in echo_valve_times.values()]
        keys = [k for k in echo_valve_times.keys()]
        ax.set_xticks(vals)
        ax.set_xticklabels(keys)
        
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_single_regional_gamma(gammas, path, valve_times = None,
                               include_global =False):

    if len(gammas) == 0: return
    
    setup_plot()

    m = len(gammas)
    colors = get_colormap(m)#[cmap(i) for i in np.linspace(0, 1, m)]
    linestyles = lineStyles*(int(m/3.)+1)
    fig = plt.figure()
    ax = fig.gca()
    

    if include_global:
        labels = ["global"] + ["region {}".format(i+1) for i in range(m-1)]
    else:
        if m == 3:
            labels = ["LV", "Septum", "RV"]
        elif m == 2:
            labels = ["LV", "RV"]
        else:
            labels =  ["region {}".format(i+1) for i in range(m)]
        

        
    for i, g in enumerate(gammas):
        n =  len(g)
        x = range(n)
        
        ax.plot(x, g, label = labels[i],
                color = colors[i], linestyle = linestyles[i])

    
    ax.set_xlabel("Point")
    ax.set_ylabel(r"$\overline{\gamma}$")
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    
    if valve_times is not None:
        echo_valve_times = valve_times["echo_valve_time"]
        pfb = valve_times["passive_filling_begins"]
        vals = [(a-pfb)%n for a in echo_valve_times.values()]
        keys = [k for k in echo_valve_times.keys()]
        ax.set_xticks(vals)
        ax.set_xticklabels(keys)

        
    fig.savefig(path, bbox_extra_artists=(lgd,),
                bbox_inches='tight', format = "png")
    print("Saved to {}".format(path))
    
def plot_single_pv_loop(v_sim, v_meas, pressure, path, unload =False):

    
    setup_plot()
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(v_sim, pressure[:len(v_sim)], "k-", label = "simulated")
    v = v_meas[1:] if unload else v_meas
    P = pressure[1:len(v)+1] if unload else pressure[:len(v)]
    
    ax.plot(v, P, "ro", label = "measured")
    ax.set_xlabel("Volume (ml)")
    ax.set_ylabel(r"Pressure (kPa)")
    ax.legend()

    fig.savefig(path, format="png")
    print("PV loop plot saved to {}".format(os.path.abspath(path)))

    
def plot_multiple_pv_loop(vs_sim, vs_meas, pressures, path, labels):

    msg = "Lists of data are of different sizes"
    assert len(vs_sim) == len(vs_meas) == len(pressures), msg
    setup_plot()
    colors = get_colormap(len(vs_sim))
    linestyles = ["s", "^"]
    
    fig = plt.figure()
    ax = fig.gca()
    for i, (v_sim, v_meas, pressure) in enumerate(zip(vs_sim, vs_meas, pressures)):

        ax.plot(v_meas, pressure[:len(v_meas)], color = colors[i], linestyle="-", label = labels[i] + " (measured)")
        ax.scatter(v_sim, pressure[:len(v_sim)], color = "k", marker = linestyles[i], label = labels[i] + " (simulated)")

    ax.set_xlabel("Volume (ml)",  fontsize = 11)
    ax.set_ylabel("Pressure (kPa)",  fontsize = 11)
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.legend(loc = "best")
    # fig.savefig(path)
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_pv_loop_w_elastance(vs_sim, vs_meas, pressures, es, v0s, ES, path, labels):
    
    msg = "Lists of data are of different sizes"
    assert len(vs_sim) == len(vs_meas) == len(pressures), msg
    setup_plot()
    colors = get_colormap(len(vs_sim))
    linestyles = ["s", "^"]
    
    fig = plt.figure()
    ax = fig.gca()
    for i, (v_sim, v_meas, pressure, e, v0) in enumerate(zip(vs_sim, vs_meas, pressures, es, v0s)):

        if i == 0:
            ax.plot([v0[ES[i]], v_sim[ES[i]]], [0,pressure[ES[i]]], "g-", label = "ESPVR")
        else:
            # No label
            ax.plot([v0[ES[i]], v_sim[ES[i]]], [0,pressure[ES[i]]], "g-")
            
        ax.plot(v_meas, pressure[:len(v_meas)], color = colors[i], linestyle="-", label = labels[i] + " (measured)")
        ax.scatter(v_sim, pressure[:len(v_sim)], color = "k", marker = linestyles[i], label = labels[i] + " (simulated)")

    ax.set_xlabel("Volume (ml)")
    ax.set_ylabel("Pressure (kPa)")
    # ax.legend(loc = "best")
    # fig.savefig(path)
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')



def plot_single_strain_curves(simulated, measured, path, groups = None, unload = False):

    setup_plot()

    colors = get_colormap(len(simulated.keys()))
    linestyles = ["s", "^"]


    fig = plt.figure()
    ax = fig.gca()
    
    
    for i,k in enumerate(simulated.keys()):

        s = simulated[k]
        if unload:
            s = s[1:]

            
        m = measured[k]
        label = k if groups is None else groups[k]
        x = range(len(s))

        ax.scatter(x,s, color = "k", marker = linestyles[i], label = label + " (simulated)")
        ax.plot(x,m, color = colors[i], label = label + " (measured)")

    # ax.legend(loc = "best")
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xlabel("Frame number", fontsize = 11)
    ax.set_ylabel("Longitudinal strain", fontsize = 11)
    # fig.savefig(path)
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')

    
    
    
def plot_strains(simulated_strains, measured_strains, outdir, dirs = None,
                 nregions = None, unload=False, name = "strain", scale = True):

    import numpy as np
    setup_plot()

    
    paths = []
    if has_seaborn:
        sns.set_style("ticks")
        sns.set_context("paper")
   
    ## Put the strain dictionaries in arrays
    
    # Basal, Mid, Apical

    
    dirs = ['circumferential','radial', 'longitudinal'] if dirs is None else dirs

    if not simulated_strains.has_key(dirs[0]):

        # Then we assume that this dictionary dont have this level and therefore
        # the next level is the regions
     
        simulated_strains = {dirs[0]:simulated_strains}
        measured_strains = {dirs[0]:measured_strains}

        
    nregions = len(simulated_strains[dirs[0]].keys()) if nregions is None else nregions
    
    big_labels = [d.title() for d in dirs]
    small_labels = [r"Basal", r"Mid", r"Apical"]
    labels = [r"Measured", r"Simulated"]

    regions = {1:"Anterior",
               2:"Septum", 
               3:"Inferior", 
               4:"Lateral",
               5:"Posterior",
               6:"Anteroseptal",
               7:"Apex"}


    if nregions == 16:
        regions_sep = [[1,3,4,6,5,2], [7,9,10,12,11,8], range(13, 17)]
        grid = range(1,22)

    elif nregions == 18:
        regions_sep = [[1,3,4,6,5,2], [7,9,10,12,11,8], range(13, 19)]
        grid = range(1,22)
        
    elif nregions == 12:
        regions_sep = [[1,3,4,6,5,2], [7,9,10,12,11,8]]
        small_labels = [r"Basal", r"Mid"]
        grid = range(1,15)
        regions.pop(7)
        
    else: # nregions = 17
        regions_sep = [[1,3,4,6,5,2], [7,9,10,12,11,8], range(13, 18)]
        grid = range(1,22)
   
    
    for d in range(len(dirs)):
        direction = dirs[d]
        strains = []
        
        smaxs = []
        smins = []
        for i in range(len(regions_sep)):

            smax = -np.inf 
            smin = np.inf
            for region in regions_sep[i]:
                    
                sim_region = str(region) if isinstance(simulated_strains[direction].keys()[0], str) \
                             else region
                meas_region = str(region) if isinstance(measured_strains[direction].keys()[0], str) \
                              else region


                if unload:
                    cur_strain = [measured_strains[direction][meas_region],
                                  simulated_strains[direction][sim_region][1:]]
                else:
                    cur_strain = [measured_strains[direction][meas_region],
                                  simulated_strains[direction][sim_region]]

                strains.append(cur_strain)

               
                min_strain = np.min([np.min(measured_strains[direction][meas_region]),
                                     np.min(simulated_strains[direction][sim_region])])
                max_strain = np.max([np.max(measured_strains[direction][meas_region]),
                                     np.max(simulated_strains[direction][sim_region])])

                smax = max_strain if max_strain > smax else smax
                smin = min_strain if min_strain < smin else smin

            smaxs.append(smax)
            smins.append(smin)
   
        fig, big_ax = plt.subplots(figsize=(15.0, 6.0))

        # Labels for the big figure

        # Turn off axis lines and ticks of the big subplot
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        # removes the white frame
        big_ax.set_frame_on(False)

        # Set the labels 
        big_ax.set_ylabel(big_labels[d], fontsize = 32)
        big_ax.yaxis.set_label_position("right")
        big_ax.yaxis.labelpad = 20


        
        # % of cardiac cycle
        x = np.linspace(0,100, len(strains[0][0]))

    
        t = 0
        # Add subplots with strain plots
        for i in grid:

            if 1:
                ax = fig.add_subplot(3,7,i)

                # Put titles on the top ones at each level
                if d == 0:
                    if i in regions.keys():
                        ax.set_title(r"{}".format(regions[i]) , fontsize = 28, y = 1.1)

                if i in [7, 14, 19, 20]:
                    ax.set_axis_off()
                    continue

                # Put ticks on every one of them
                ax.set_xlim(0,100)
                ax.set_xticks([0,50,100])
             
                if scale:
                    if i <= 7:
                        ax.set_ylim(smins[0], smaxs[0])
                        ax.set_yticks([smins[0], 0, smaxs[0]])
                    elif i > 14:
                        ax.set_ylim(smins[2], smaxs[2])
                        ax.set_yticks([smins[2], 0, smaxs[2]])

                    else:
                        ax.set_ylim(smins[1], smaxs[1])
                        ax.set_yticks([smins[1], 0, smaxs[1]])

                # Put xlabels only on the bottom ones
                if i in [12,13,15, 16, 17, 18, 21] and d == 2: 
                    ax.set_xticklabels([0,50,100], fontsize = 22)
 
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])

                # Put y labels only on the left most ones
                if scale:
                    if i not in [1,8,15]:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])
                    else:

                        ax.set_yticklabels([smin, 0, smax], fontsize = 22)
                        if i == 1:
                            ax.set_ylabel(small_labels[0], fontsize = 28) # Basal
                        elif i == 8:
                            ax.set_ylabel(small_labels[1], fontsize = 28) # Mid
                        else:
                            ax.set_ylabel(small_labels[2], fontsize = 28) # Apical

                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

                try:
                    strain = strains[t]
                    t+= 1
                except IndexError:
                    pass
                
                else:

                    n0 = len(strain[0])
                    n1 = len(strain[1])
                
                    l1 = ax.plot(x[:n0], strain[0], "b-",label = labels[0])
                    l2 = ax.plot(x[:n1], strain[1], "r-", label = labels[1])
                    
                    ax.axhline(y=0, ls = ":")
                    lines = [l1[0], l2[0]]
                    


            # Plot the legend
            if i == 7:
                ax = fig.add_subplot(3,7,21)
                ax.set_axis_off()
                ax.legend(lines, labels, "center", prop={'size':20})

        # Adjust size
        fig.tight_layout(w_pad = 0.0)
     
        # Remove top and right axis
        if has_seaborn:
            sns.despine()
        path = "{}/simulated_{}_{}.pdf".format(outdir, name, direction)
        paths.append(path)
        fig.savefig(path,
                    bbox_inches='tight')
        plt.close()
        
    return paths
def plot_strain_scatter(data, path, labels = None, split = 0):

    assert split in [0,1,2]
    
    setup_plot()
    
    fig = plt.figure()
    ax = fig.gca()
    
    ax.set_ylabel("Simulated strain")
    ax.set_xlabel("Measured strain")


    if split == 0:
        color = get_colormap(1)
        assert np.all(np.sort(data.keys()) == \
                      np.sort(["simulated", "measured"]))
        
        ax.scatter(data["measured"],data["simulated"], color = color)
        smin = np.min([np.min(data["simulated"]),
                       np.min(data["measured"])])
        smax = np.max([np.max(data["simulated"]),
                       np.max(data["measured"])])
        ax.plot([smin, smax], [smin,smax], "k-")
        fig.savefig(path)
        
    elif split == 1:

        
        assert np.all(np.sort(data.keys()) == \
                      np.sort(["simulated_passive", "measured_passive",
                               "simulated_active", "measured_active"]))
        colors = get_colormap(2)
        labels = {"passive":"passive", "active":"active"} \
                 if labels is None else labels
        
        s_min = np.inf
        s_max = -np.inf
        fits = {}
        
        msg = "#"*40 + "\nTesting Strain Scatter.\n" + \
              "Copmuting the peason r \n "
        print(msg)


        n_measurements = 0
        for i, key in enumerate(["active", "passive"]):


            ax.scatter(data["measured_{}".format(key)],
                       data["simulated_{}".format(key)],
                       label = labels[key], color = colors[i])


            # Do some statistics
            x = data["measured_{}".format(key)]
            y = data["simulated_{}".format(key)]

            n_measurements += len(x)

            # slope, intercept, r_value, p_value, std_err
            fits[key] = stats.linregress(x, y)
       
            msg = """{}: \n\tR^2 = {}\n\tSlope = {}\n\tIntercept = {} 
            \n\tR = {}\n\tP-value = {}\n\tStd Errror = {} \n""".format(key, fits[key][2]**2, *fits[key])
            print(msg)
                                                                               
            s_min = np.min([s_min,
                            np.min(data["measured_{}".format(key)]),
                            np.min(data["simulated_{}".format(key)])])

            s_max = np.max([s_max,
                            np.max(data["measured_{}".format(key)]),
                            np.max(data["simulated_{}".format(key)])])

        print("#"*40)                                                       
        # Plot the 1-1 line
        x = np.array([s_min, s_max])
        sign = {-1:"-", 1:"+"}

        for i, (k, v) in enumerate(fits.iteritems()):

            if k == "passive":
                label = "diastolic fit, $R^2 = {:.2f}$".format(v[2]**2)
            else:
                label = "systolic fit, $R^2 = {:.2f}$".format(v[2]**2)
                
            ax.plot(x, v[0]*x +v[1],
                    color = "k", linestyle = lineStyles[i],
                    label = label)
            
                    
                    # label = "{0:.2f}x {1} {2:.1e}".format(v[0],
                    #                                      sign[np.sign(v[1])],
                    #                                      np.abs(v[1])))

        msg = "\nNumber of volume meaurements = {}\n".format(n_measurements)
        print(msg)
        # ax.plot(x,x, "k-")
        # axbox = ax.get_position()
        ax.set_ylim([-0.5, 1.0])
        ax.legend(loc = "upper left")
        # ax.legend(loc = (axbox.x0 + 0.2, axbox.y0 + 0.7 ))#"upper left")
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        #           fancybox=True, shadow=False, ncol=2)
        fig.savefig(path)
        # lgd = ax.legend(loc = "center left",
                        # bbox_to_anchor=(1, 0.5))
        # fig.savefig(path, bbox_extra_artists=(lgd,),
                    # bbox_inches='tight')
        
    else:

        
        assert np.all(np.sort(data.keys()) == \
                      np.sort(["simulated_passive", "measured_passive",
                               "simulated_active", "measured_active"]))

        for key in data.keys():
            assert np.all(np.sort(data[key].keys()) == \
                          np.sort(["longitudinal", "circumferential", "radial"]))

        # colors = [cmap(i) for i in np.linspace(0, 1, 3)]
        colors = get_colormap(3)
        for i, key in enumerate(["longitudinal", "circumferential", "radial"]):
            ax.scatter(data["measured_passive"][key],
                       data["simulated_passive"][key],
                       label = key + " (passive)",
                       marker = markers[0], color = colors[i])
            
            ax.scatter(data["measured_active"][key],
                       data["simulated_active"][key],
                       label = key + " (active)",
                       marker = markers[1], color = colors[i])

            
            
        lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
        fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_volume_scatter(simulated, measured, path):

    setup_plot()

    #Do some statistics
    msg = "#"*40 + "\nTesting Volume Scatter.\n" + \
          "Copmuting the peason r \n "
    print(msg)
    fit = stats.linregress(measured,simulated)
       
    msg = """\n\tR^2 = {}\n\tSlope = {}\n\tIntercept = {} 
    \n\tR = {}\n\tP-value = {}\n\tStd Errror = {} \n""".format(fit[2]**2, *fit)
    print(msg)
    print("#"*40) 

    # Plot
    fig = plt.figure()
    ax = fig.gca()
   
    
    ax.set_ylabel("Simulated volume (ml)")
    ax.set_xlabel("Measured volume (ml)")
    ax.scatter(measured, simulated, color = "b", label = "volume")

    msg = "\nNumber of volume meaurements = {}\n".format(len(measured))
    print(msg)
    vmin = np.min([np.min(simulated), np.min(measured)])
    vmax = np.max([np.max(simulated), np.max(measured)])
    x = np.array([vmin, vmax])

    sign = {-1:"-", 1:"+"}
        
    ax.plot(x, fit[0]*x +fit[1],"k-",
            label = "linear fit,  $R^2 = {:.2f}$".format(fit[2]**2))
            # label = "{0:.2f}x {1} {2:.1e}".format(fit[0],
            #                                       sign[np.sign(fit[1])],
            #                                       np.abs(fit[1])))
    

    
    # ax.plot([vmin, vmax], [vmin,vmax], "k-")

    ax.legend(loc = "upper left")
    fig.savefig(path)

    
def plot_emax(emax, labels, path):

    setup_plot()
    fig = plt.figure()
    ax = fig.gca()

    ax.set_yscale('log')
    
    width = 1.0/len(emax)
    hatches = ["", "//"]*(len(emax)/2)
    xticks = []
    
    for i, e in enumerate(emax, start = 1):
        m = np.mean(e)
        s = np.std(e)
        
        ax.bar(i, m, width, fill = False,
               linewidth = 2, edgecolor='black', hatch=hatches[i-1])#,
               # yerr=s, error_kw=dict(elinewidth=2,ecolor='black'))
   
        if not np.isscalar(e):
            ax.scatter([i+0.5*width]*len(e), e, color="k", s = 18)
        xticks.append(i +width*0.5)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\tilde{E}_{\mathrm{ES}}$ (kPa/ml)")
    fig.savefig(path)
def plot_time_varying_elastance(volumes, pressures, elastances, v0s, path):

    setup_plot()
    
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(volumes, pressures)
    for i, (p,v) in enumerate(zip(pressures, volumes)):
        
        ax.plot([v0s[i], v], [0,p], "k-")
        # ax.annotate("{:.2f}".format(elastances[i]), (v,p), size = 6)
        
    ax.set_xlabel("Volume (ml)")
    ax.set_ylabel("Pressure (kPa)")
    
    fig.savefig(path)

def plot_cardiac_work(work, labels, measured_work, path):
    
    setup_plot()

    m = len(work)+1
    colors = [cmap(i) for i in np.linspace(0, 1, m)]
    linestyles = lineStyles*(int(m/3.)+1)
    fig = plt.figure()
    ax = fig.gca()


    for i,w in enumerate(work):
        n = len(w)
        x = range(n)

        ax.plot(x, w, label = labels[i],
                color = colors[i], linestyle = linestyles[i])
        
    ax.plot(x, measured_work[:n], "k-", label = "Measured")

    ax.set_xlabel("Point")
    ax.set_ylabel(r"Cardiac Work (Joule/m3)")
    lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
    
    fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_geometric_distance(dist, labels, path, valve_times = None, groups = None, stds = None):

    setup_plot()

    colors = get_colormap(len(dist))
    color_group = None if groups is None else get_colormap(len(groups.keys()))
    labels = [""]*len(dist) if labels is None else labels
    n = len(dist[0])
    x = range(n)

    
    fig = plt.figure()
    ax = fig.gca()

    
    if groups:
        dist_groups = [[] for i in range(len(groups.keys()))]
        labels_groups = groups.keys()
        
        for i, (k,v) in enumerate(groups.iteritems()):
    
            label_on = False
            for j in range(len(dist)):
               
                if labels[j] in v:
               
                    if not label_on:
                        ax.plot(x,dist[j], label = k, color = color_group[i])
                        label_on = True
                    else:
                        ax.plot(x,dist[j], color = color_group[i])
                     

    else:
            
        for i, d in enumerate(dist):
                
            ax.plot(x, d, label = labels[i], color = colors[i])
            if stds is not None:
                ax.fill_between(x, d + stds[i], d - stds[i],
                                facecolor = colors[i], alpha = 0.5)
            
    if valve_times:

        echo_valve_times = valve_times["echo_valve_time"]
        pfb = valve_times["passive_filling_begins"]
        # vs = ["mvc", "avo", "avc", "mvo"]
        # [echo_valve_times[v] for v in vs]
        vals = [(a-pfb)%n for a in echo_valve_times.values()]
        keys = [k for k in echo_valve_times.keys()]


        ax.set_xticks(vals)
        ax.set_xticklabels(keys, rotation = 45)

            
    ax.set_xlabel("Valvular event")
    ax.set_ylabel(r"Distance to segmentation (cm)")
    
    if len(labels) > 2:
        lgd = ax.legend(loc = "center left", bbox_to_anchor=(1, 0.5))
        fig.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        ax.legend(loc = "upper right")
        fig.savefig(path)
