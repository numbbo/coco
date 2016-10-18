# -*- coding: utf-8 -*-
"""
 Called by plots_alongDirections and doing the actual plotting of aRT values
 to attain all objective vectors in a certain interval [ideal, c*nadir] with
 c being the constant `maxplot` defined below.

 based on code by Thanh-Do Tran 2012--2015
 adapted by Dimo Brockhoff 2016
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np  # "pip install numpy" installs numpy
import os
import sys
import colorsys
from itertools import product

from bbob_pproc.ppfig import saveFigure
import bbobbenchmarks as bm


decimals=2 # precision for downsampling
maxplot = 10 # maximal displayed value (assuming nadir in [1,1])
precision = 1e-3 # smallest displayed value in logscale
maxbudget = '1e6 * dim'  # largest budget for normalization of aRT-->sampling conversion
minbudget = '1'          # smallest budget for normalization of aRT-->sampling conversion
cropbudget = maxbudget   # objective vectors produced after cropbudget not taken into account
n = 100 # number of grid points per objective
grayscale = False

biobjinst = {1: [2, 4],
             2: [3, 5],
             3: [7, 8],
             4: [9, 10],
             5: [11, 12],
             6: [13, 14],
             7: [15, 16],
             8: [17, 18],
             9: [19, 21],
             10: [21, 22]}


def generate_aRTA_plot(f_id, dim, f1_id, f2_id,
                   outputfolder="./", inputfolder=None, tofile=True,
                   logscale=True, downsample=True, with_grid=False):
    """
    Objective Space plot, indicating for each (grid) point
    the runtime of the algorithm to attain it.

    Takes into account the corresponding COCO archive files in
    the given outputfolder

    Assumes that each instance is only contained once in the
    data.
    """
    
    [gridpoints, aRTs, A] = get_all_aRT_values_in_objective_space(f_id,
                                dim, f1_id, f2_id, inputfolder=inputfolder,
                                downsample=downsample, with_grid=with_grid)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

#    # print all (downsampled) points of all runs
#    for key in A:
#        for a in A[key]:
#            plt.plot(a[1], a[2], 'xk')

#    for a in A[9]:
#        plt.plot(a[1], a[2], 'ob')
    

#    for g in gridpoints:
#        plt.plot(g[0], g[1], '+m')


    # normalize colors:
    logcolors = np.log10(aRTs)
    logcolors = (logcolors - np.log10(eval(minbudget)))/(np.log10(eval(maxbudget))-np.log10(eval(minbudget)))

    if grayscale:
        aRTA_colormap = matplotlib.cm.gray_r
    else:
        aRTA_colormap = matplotlib.cm.hot_r
        #aRTA_colormap = matplotlib.cm.nipy_spectral_r

    for i in range(len(gridpoints)-1, -1, -1):
    #for i in range(1, len(gridpoints)-3, 1):
        if not np.isfinite(logcolors[i]):
            continue # no finite aRT
        ax.add_artist(patches.Rectangle(
                ((gridpoints[i])[0], (gridpoints[i])[1]),
                 maxplot-(gridpoints[i])[0],
                 maxplot-(gridpoints[i])[1],
                 alpha=1.0,
                 color=aRTA_colormap(logcolors[i])))
            
    # Add a single point outside of the axis range with the same cmap and norm
    axscat = plt.scatter([-100], [-100], c=[0], cmap=aRTA_colormap)
    axscat.set_clim([0, 100])                # mainly to set correct tick values
    cb = plt.colorbar(ticks=[0, 33, 66, 100])   # mainly to set correct tick values
    cb.ax.set_yticklabels(['1', '1e2*n',        # attention: might be wrong
                           '1e4*n', '1e6*n'])  # if minbudget or maxbudget
                                                 # are changed !!!!!
    
    # beautify:
    ax.set_title("aRT in objective space for bbob-biobj function $f_{%d}$ (%d-D)" % (f_id, dim))
    [line.set_zorder(3) for line in ax.lines]
    [line.set_zorder(3) for line in ax.lines]
    #fig.subplots_adjust(left=0.1) # more room for the y-axis label
    if logscale:                
        ax.set_xlabel(r'log10($f_1 - f_1^\mathsf{opt}$) (normalized)', fontsize=16)
        ax.set_ylabel(r'log10($f_2 - f_2^\mathsf{opt}$) (normalized)', fontsize=16)    
        # we might want to zoom in a bit:
        ax.set_xlim(precision, maxplot)
        ax.set_ylim(precision, maxplot)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlabel(r'$f_1 - f_1^\mathsf{opt}$ (normalized)', fontsize=16)
        ax.set_ylabel(r'$f_2 - f_2^\mathsf{opt}$ (normalized)', fontsize=16)    
        # we might want to zoom in a bit:
        ax.set_xlim((0, maxplot))
        ax.set_ylim((0, maxplot))
    
    
    # printing
    if tofile:
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        filename = outputfolder + "aRTobjspace-f%02d-d%02d" % (f_id, dim)
        if logscale:
            filename = filename + "-log"
        saveFigure(filename)
    else:        
        plt.show(block=True)

    plt.close()
    

def get_all_aRT_values_in_objective_space(f_id, dim, f1_id, f2_id,
                   inputfolder=None, logscale=True, downsample=True,
                   with_grid=False):
    """
    Returns a set of points in objective space and their corresponding
    aRT values for the specified algorithm data (on function `f_id` in
    dimension `dim`, given in the folder `inputfolder`). Data points
    produced after cropbudget function evaluations are not taken into account.
    
    The points in objective space are thereby either generated on a grid
    (if `with_grid == True` either in logscale or not) or constructed from the
    actual data points of the algorithm (TODO: not supported yet). Note that
    the points will be already sorted in order of their aRTs.
    
    If `downsample == True`, the input data will be reduced by taking into
    account only one input point per objective space cell where the cells
    are given by cutting the objective vectors to the given number of
    `decimals` decimals. For later plotting of the input points, the
    already downsampled input points are also returned as a third argument
    (in a dictionary, giving for each entry the data points associated to
    the corresponding instance/run).

    Assumes that each instance is only contained once in the data.
    """
    
    
    # obtain the data of the algorithm run to display:
    filename = "bbob-biobj_f%02d_d%02d_nondom_all.adat" % (f_id, dim)
    try:
        A = {}
        instance = -1
        B = []
        nadirs = {}
        ideals = {}
        with open(inputfolder + filename) as f:
            for line in f:
                if "function eval_number" in line:
                    continue
                elif "instance" in line:
                    # store first data of previous instance:
                    if instance not in A and not instance == -1:
                        # downsample, i.e., filter out all but one point per grid cell in the 
                        # objective space
                        blen = len(B)
                        if downsample:
                            B = sample_down(B, decimals)
                        print("instance data points downsampled from %d to %d" % (blen, len(B)))
                        
                        A[instance] = B

                    # reset instance and B:
                    instance = int((line.split()[3])[:-1])
                    B = []
                    # get ideal and nadir for this instance:
                    f1, f1opt = bm.instantiate(f1_id, iinstance=biobjinst[instance][0])
                    f2, f2opt = bm.instantiate(f2_id, iinstance=biobjinst[instance][1])
                    fdummy = f1.evaluate(np.zeros((1, dim)))
                    fdummy = f2.evaluate(np.zeros((1, dim)))
                    nadir = np.array([f1.evaluate(f2.xopt), f2.evaluate(f1.xopt)])
                    ideal = np.array([f1opt, f2opt])
                    nadirs[instance] = nadir
                    ideals[instance] = ideal
                else:
                    splitline = line.split()
                    newline = np.array(splitline[:3], dtype=np.float)
                    if newline[0] <= eval(cropbudget):
                        # normalize objective vector:
                        newline[1] = (newline[1]-ideals[instance][0])/(nadirs[instance][0]-ideals[instance][0])
                        newline[2] = (newline[2]-ideals[instance][1])/(nadirs[instance][1]-ideals[instance][1])

                        B.append(newline)
                    
            # store data of final instance:
            if instance not in A and not instance == -1:
                blen = len(B)
                if downsample:
                    B = sample_down(B, decimals)
                A[instance] = B
                print("instance data points downsampled from %d to %d" % (blen, len(B)))

            print("all %d instances read in" % len(A))


    except:
        print("Problem opening %s" % (inputfolder + filename))
        e = sys.exc_info()[0]
        print("   Error: %s" % e)

    
    # construct grid in normalized objective (sub-)space [precision, maxplot]:
    if with_grid:
        if logscale:
            log_range = np.logspace(np.log10(precision), np.log10(maxplot), num=n, endpoint=True, base=10.0)
            gridpoints = np.array(list(product(log_range, log_range)))
        else:
            gridpoints = maxplot * np.array(list(product(range(n),range(n))))/(n-1)
    else:
        raise NotImplementedError # there is currently a bug in the code!!!
        
        gridpoints = []
        for key in A:
            for a in A[key]:
                gridpoints.append([a[1], a[2]]) # extract only objective vector
        gridpoints = np.array(gridpoints)


    aRTs = compute_aRT(gridpoints, A)

    # sort gridpoints (and of course colors) wrt. their aRT:
    idx = aRTs.argsort(kind='mergesort')
    aRTs = aRTs[idx]
    gridpoints = gridpoints[idx]

    
    return gridpoints, aRTs, A
    


def generate_aRTA_ratio_plot(f_id, dim, f1_id, f2_id,
                   outputfolder="./", inputfolder_1=None, 
                   inputfolder_2=None, tofile=True,
                   logscale=True, downsample=True):
    """
    Objective Space plot, showing the aRT ratios between two algorithms,
    given in `inputfolder_1` and `inputfolder_2` for each point on a grid in
    objective space.

    Assumes that each instance is only contained once in the
    data.
    """
    
    [gridpoints, aRTs_1, A] = get_all_aRT_values_in_objective_space(f_id,
                                    dim, f1_id, f2_id, inputfolder=inputfolder_1,
                                    downsample=downsample, with_grid=True)
    print('Computing aRT values for %s done.' % inputfolder_1)

    [gridpoints_2, aRTs_2, A] = get_all_aRT_values_in_objective_space(f_id,
                                    dim, f1_id, f2_id, inputfolder=inputfolder_2,
                                    downsample=downsample, with_grid=True)
    print('Computing aRT values for %s done.' % inputfolder_2)

    # resort gridpoints from lower left to top right in order to not loose
    # the right color by overlapping rectangles...
    # assuming that gridpoints and gridpoints_2 are the same
    idx = gridpoints[:,1].argsort(kind='mergesort')
    gridpoints = gridpoints[idx]
    aRTs_1 = aRTs_1[idx]
    aRTs_2 = aRTs_2[idx]
    idx = gridpoints[:,0].argsort(kind='mergesort')
    gridpoints = gridpoints[idx]
    aRTs_1 = aRTs_1[idx]
    aRTs_2 = aRTs_2[idx]

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    
    # compute ratios (in favor of each algorithm: i.e. positive if ratio>1
    # in favor of Algorithm A and negative if ratio>1 in favor of Algorithm B)
    ratio_in_favor_of_Alg1 = np.maximum(aRTs_2 / aRTs_1, 1.)
    ratio_in_favor_of_Alg2 = np.maximum(aRTs_1 / aRTs_2, 1.)

    aRT_ratios = np.zeros(len(ratio_in_favor_of_Alg1))
    for i in range(len(ratio_in_favor_of_Alg1)):
        if np.isfinite(ratio_in_favor_of_Alg1[i]) and np.isfinite(ratio_in_favor_of_Alg2[i]):
            aRT_ratios[i] = ratio_in_favor_of_Alg1[i] if ratio_in_favor_of_Alg1[i] > 1 else 0
            aRT_ratios[i] = - ratio_in_favor_of_Alg2[i] if ratio_in_favor_of_Alg2[i] > 1 else aRT_ratios[i]
        elif not np.isfinite(ratio_in_favor_of_Alg1[i]) and np.isfinite(ratio_in_favor_of_Alg2[i]):
            aRT_ratios[i] = - np.inf
        elif np.isfinite(ratio_in_favor_of_Alg1[i]) and not np.isfinite(ratio_in_favor_of_Alg2[i]):
            aRT_ratios[i] = np.inf
        else: # both aRT values are infinite
            aRT_ratios[i] = 0

    norm = matplotlib.colors.Normalize(vmin=-10.,vmax=10., clip=False)
    if grayscale:
        aRTA_colormap = matplotlib.cm.gray_r
    else:
        aRTA_colormap = matplotlib.cm.RdBu


    # Add a single point outside of the axis range with the same cmap and norm
    axscat = plt.scatter([-100], [-100], c=[0], cmap=aRTA_colormap)
    axscat.set_clim([-10, 10])                # mainly to set correct tick values

    for i in range(len(gridpoints)):
    #for i in range(1, len(gridpoints)-3, 1):
        if not np.isfinite(aRT_ratios[i]):
            if np.isfinite(aRTs_1) and not np.isfinite(aRTs_2):
                ax.add_artist(patches.Rectangle(
                    ((gridpoints[i])[0], (gridpoints[i])[1]),
                     maxplot-(gridpoints[i])[0],
                     maxplot-(gridpoints[i])[1],
                     alpha=1.0,
                     color='green'))
                print('green')
            if not np.isfinite(aRTs_1) and np.isfinite(aRTs_2):
                ax.add_artist(patches.Rectangle(
                    ((gridpoints[i])[0], (gridpoints[i])[1]),
                     maxplot-(gridpoints[i])[0],
                     maxplot-(gridpoints[i])[1],
                     alpha=1.0,
                     color='magenta'))
                print('magenta')
            continue # no finite aRT for >= 1 algo
        ax.add_artist(patches.Rectangle(
                ((gridpoints[i])[0], (gridpoints[i])[1]),
                 maxplot-(gridpoints[i])[0],
                 maxplot-(gridpoints[i])[1],
                 alpha=1.0,
                 color=aRTA_colormap(norm(aRT_ratios[i]))))
#        ax.plot((gridpoints[i])[0], (gridpoints[i])[1], 's', color=aRTA_colormap(norm(aRT_ratios[i])), markersize=18)
    
    cb = plt.colorbar(ticks=[-10, -5, 0, 5, 10])  # mainly to set correct tick values
    cb.ax.set_yticklabels(['10', '5', '0', '5', '10'])
    cb.set_label("<-- in favor of Algo B      aRT ratio      in favor of Algo A -->")
    
    # beautify:
    ax.set_title("aRT ratios in objective space for bbob-biobj function $f_{%d}$ (%d-D)" % (f_id, dim))
    [line.set_zorder(3) for line in ax.lines]
    [line.set_zorder(3) for line in ax.lines]
    #fig.subplots_adjust(left=0.1) # more room for the y-axis label
    if logscale:                
        ax.set_xlabel(r'log10($f_1 - f_1^\mathsf{opt}$) (normalized)', fontsize=16)
        ax.set_ylabel(r'log10($f_2 - f_2^\mathsf{opt}$) (normalized)', fontsize=16)    
        # we might want to zoom in a bit:
        ax.set_xlim(precision, maxplot)
        ax.set_ylim(precision, maxplot)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlabel(r'$f_1 - f_1^\mathsf{opt}$ (normalized)', fontsize=16)
        ax.set_ylabel(r'$f_2 - f_2^\mathsf{opt}$ (normalized)', fontsize=16)    
        # we might want to zoom in a bit:
        ax.set_xlim((0, maxplot))
        ax.set_ylim((0, maxplot))
    
    
    # printing
    if tofile:
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        filename = outputfolder + "aRT-ratios-objSpace-f%02d-d%02d" % (f_id, dim)
        if logscale:
            filename = filename + "-log"
        saveFigure(filename)
    else:        
        plt.show(block=True)

    plt.close()

    
    
def compute_aRT(points, A):
    """ Computes the average runtime to attain the objective vectors in points
        by the algorithm, with algorithm data given in dictionary A).
        
        Assumes that the algorithm data in A is given in the order of
        increasing number of function evaluations for each entry.
        
        >>> A = {0: [[1, 1, 1], [3, 0.75, 0.5], [7, 0.5, 0.6]],
        ... 1: [[1, 0.9, 0.9], [2, 0.5, 0.4]]}
        >>> gridpoints = [[0.6, 0.5]]
        >>> compute_aRT(gridpoints, A)
        array([ 9.])
    """
    sum_runtimes_successful = np.zeros(len(points))
    num_runtimes_successful = np.zeros(len(points))
    sum_runtimes_unsuccessful = np.zeros(len(points))
    
    for key in A:
        points_finished = [False] * len(points)
        runtime_to_attain_points = [np.nan] * len(points)
        max_runtimes = np.zeros(len(points))
        for a in A[key]:
            for i in range(len(points)):
                if not points_finished[i]:
                    if weakly_dominates([a[1], a[2]], points[i]):
                        runtime_to_attain_points[i] = a[0]
                        points_finished[i] = True
                    else:
                        max_runtimes[i] = a[0]                        
            if min(points_finished): # all grid points dominated
                break
        for i in range(len(points)):
            if runtime_to_attain_points[i] is np.nan:
                sum_runtimes_unsuccessful[i] = sum_runtimes_unsuccessful[i] + max_runtimes[i]
            else:
                sum_runtimes_successful[i] = sum_runtimes_successful[i] + runtime_to_attain_points[i]
                num_runtimes_successful[i] = num_runtimes_successful[i] + 1                
                
        
    aRT = np.zeros(len(points))
    for i in range(len(points)):
        if num_runtimes_successful[i] > 0:
            aRT[i] = (sum_runtimes_unsuccessful[i] + sum_runtimes_successful[i])/num_runtimes_successful[i]
        else:
            aRT[i] = np.nan
    
    return aRT
    
def weakly_dominates(a,b):
    """ Returns True iff a dominates b wrt. minimization """
    
    return (a[0] <= b[0]) and (a[1] <= b[1])
    
def sample_down(B, decimals):
    """ Samples down the solutions in B, given as (#funevals, f_1, f_2)
        entries in a list or an np.array such that only one of the solutions
        with the same objective vector is kept when they are rounded to the
        given decimal.
        
        >>> A = [[1, 2.155, 3.342],
        ...      [2, 2.171, 3.326],
        ...      [2, 2.174, 3.329],
        ...      [4, 4, 2.2]]
        >>> sample_down(A, 2)
        array([[ 1.   ,  2.155,  3.342],
               [ 2.   ,  2.171,  3.326],
               [ 4.   ,  4.   ,  2.2  ]])
        >>> sample_down(A, 1)
        array([[ 1.   ,  2.155,  3.342],
               [ 4.   ,  4.   ,  2.2  ]])

    """
    
    C = np.array(B)
    C = C[C[:, 2].argsort(kind='mergesort')] # sort wrt second objective
    C = C[C[:, 1].argsort(kind='mergesort')] # now wrt first objective
    X = np.around(C, decimals=decimals)
    # sort wrt second objective first
    idx_1 = X[:, 2].argsort(kind='mergesort')
    X = X[idx_1]
    # now wrt first objective to finally get a stable sort
    idx_2 = X[:, 1].argsort(kind='mergesort')
    X = X[idx_2]
    xflag = np.array([False] * len(X), dtype=bool)
    xflag[0] = True # always take the first point
    for i in range(1, len(X)):
        if not (X[i, 1] == X[i-1, 1] and
                X[i, 2] == X[i-1, 2]):
            xflag[i] = True
    X = ((C[idx_1])[idx_2])[xflag]
    B = X[X[:, 0].argsort(kind='mergesort')] # sort again wrt. #FEs

    return B
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()