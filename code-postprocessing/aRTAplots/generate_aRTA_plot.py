# -*- coding: utf-8 -*-
"""
 Called by plot_aRTA_function.py and plot_aRTA_ratios.py and does the actual
 reading in of COCO archive data and plotting of aRT values
 to attain all objective vectors in a certain interval [ideal, maxplot] 
 if logscale=False or in the interval [precision, maxplot] with logscale=True.

 Data might be downsampled (if downsample=True) to the precision
 10^{-decimals}. All algorithm data is furthermore cropped after
 `eval(cropbudget)` many function evaluations.
 
 Note that for the moment, only aRTA function and aRTA ratio plots that rely on
 a grid of `n \times n` points in objective space are provided.
 
 Prerequisite: the cocopp module of the COCO platform needs to be installed.
 Run 'pip install cocopp'  to install the latest version from PyPI.

 based on code by Thanh-Do Tran 2012--2015
 adapted by Dimo Brockhoff 2016
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
# matplotlib.use('TkAgg')

from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np  # "pip install numpy" installs numpy
import os
import sys
from itertools import product
import time

from cocopp.ppfig import save_figure
import bbobbenchmarks as bm


decimals=2 # precision for downsampling
maxplot = 10 # maximal displayed value (assuming nadir in [1,1])
precision = 1e-3 # smallest displayed value in logscale
maxbudget = '1e6 * dim'  # largest budget for normalization of aRT-->sampling conversion
minbudget = '1'          # smallest budget for normalization of aRT-->sampling conversion
cropbudget = maxbudget   # objective vectors produced after cropbudget not taken into account
n = 200 # number of grid points per objective
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
    the runtime of the algorithm to attain it (aka aRTA function plots).

    Takes into account the corresponding COCO archive files in
    the given outputfolder.

    Assumes for now that each instance is only contained once in the
    data.
    """
    
    [gridpoints, aRTs, A] = get_all_aRT_values_in_objective_space(f_id,
                                dim, f1_id, f2_id, inputfolder=inputfolder,
                                downsample=downsample, with_grid=with_grid,
                                logscale=logscale)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # normalize colors:
    logcolors = np.log10(aRTs)
    logcolors = (logcolors - np.log10(eval(minbudget)))/(np.log10(eval(maxbudget))-np.log10(eval(minbudget)))

    if grayscale:
        aRTA_colormap = matplotlib.cm.gray_r
    else:
        aRTA_colormap = matplotlib.cm.hot_r
        #aRTA_colormap = matplotlib.cm.nipy_spectral_r

    for i in range(len(gridpoints)-1, -1, -1):
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
    axscat.set_clim([0, 100])                 # mainly to set correct tick values
    cb = plt.colorbar(ticks=[0, 33, 66, 100]) # mainly to set correct tick values
    cb.ax.set_yticklabels(['1', '1e2*n',      # attention: might be wrong
                           '1e4*n', '1e6*n']) # if minbudget or maxbudget
                                              # are changed !!!!!
    
    # beautify:
    ax.set_title("aRTA function plot for bbob-biobj function $f_{%d}$ (%d-D)" % (f_id, dim))
    [line.set_zorder(3) for line in ax.lines]
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
        filename = outputfolder + "aRTA-f%02d-d%02d" % (f_id, dim)
        if with_grid:
            filename = filename + "-%dx%dgrid" % (n,n)
        if logscale:
            filename = filename + "-log"
        save_figure(filename)
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
    the returned points will be already sorted in order of their aRTs (in
    decreasing order).
    
    If `downsample == True`, the input data will be reduced by taking into
    account only one input point per objective space cell where the cells
    correspond to the objective vectors of the above mentioned grid.
    In any case, all points outside [0,maxplot] (and [precision, maxplot] in 
    the locscale case) are removed. For later plotting of the input points, the
    already downsampled input points are also returned as a third argument
    (in a dictionary, giving for each entry the data points associated to
    the corresponding instance/run).

    Assumes that each instance is only contained once in the data.
    """
    
    # obtain the data of the algorithm run to display:
    filename = "bbob-biobj_f%02d_d%02d_nondom_all.adat" % (f_id, dim)
    #filename = "bbob-biobj_f%02d_d%02d_nondom_instance1.adat" % (f_id, dim)
    try:
        A = {}
        instance = -1
        B = []
        nadirs = {}
        ideals = {}
        if downsample:
            print('reading in data and downsampling them to %dx%d grid...' % (n, n))
        else:
            print('reading in data...')
        
        with open(inputfolder + filename) as f:
            for line in f:
                if "function eval_number" in line:
                    continue
                elif "evaluations =" in line:
                    continue
                elif "instance" in line:
                    # store first data of previous instance:
                    if instance not in A and not instance == -1:
                        # downsample, i.e., filter out all but one point per grid cell in the 
                        # objective space
                        blen = len(B)
                        if downsample:
                            B = sample_down(B, n, logscale=logscale)
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
                        # assume that all points are >0 for both objectives
                        # and remove all above `maxplot`:
                        if newline[1] <= maxplot and newline[2] <= maxplot:
                            B.append(newline)
                            
            # store data of final instance:
            if instance not in A and not instance == -1:
                blen = len(B)
                if downsample:
                    B = sample_down(B, n, logscale=logscale)
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
        raise NotImplementedError # for the moment, the plotting is not
                                  # memory-efficient enough to handle even
                                  # small data sets
        ticks = []
        for key in A:
            for a in A[key]:
                if a[1] not in ticks:
                    ticks.append(a[1])
                if a[2] not in ticks:
                    ticks.append(a[2])
        ticks.sort()
        print("producing set of potential %dx%d (irregular) grid points where aRTA plot can change" % (len(ticks), len(ticks)))
        gridpoints = np.array(list(product(ticks, ticks)))
        


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
    Objective Space plot, showing the aRT ratios between two algorithms
    (aka aRTA ratio function), given in `inputfolder_1` and `inputfolder_2`
    for each point on a grid in objective space.

    Assumes that each instance is only contained once in the
    data.
    """
    
    [gridpoints, aRTs_1, A] = get_all_aRT_values_in_objective_space(f_id,
                                    dim, f1_id, f2_id, inputfolder=inputfolder_1,
                                    downsample=downsample, with_grid=True,
                                    logscale=logscale)
    print('Computing aRT values for %s done.' % inputfolder_1)

    [gridpoints_2, aRTs_2, A] = get_all_aRT_values_in_objective_space(f_id,
                                    dim, f1_id, f2_id, inputfolder=inputfolder_2,
                                    downsample=downsample, with_grid=True,
                                    logscale=logscale)
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
        aRTA_colormap = matplotlib.cm.gray
    else:
        aRTA_colormap = matplotlib.cm.RdBu # matplotlib.cm.spring


    # Add a single point outside of the axis range with the same cmap and norm
    axscat = plt.scatter([-100], [-100], c=[0], cmap=aRTA_colormap)
    axscat.set_clim([-10, 10])                # mainly to set correct tick values

    for i in range(len(gridpoints)):
        if not np.isfinite(aRT_ratios[i]):
            if np.isfinite(aRTs_1) and not np.isfinite(aRTs_2):
                ax.add_artist(patches.Rectangle(
                    ((gridpoints[i])[0], (gridpoints[i])[1]),
                     maxplot-(gridpoints[i])[0],
                     maxplot-(gridpoints[i])[1],
                     alpha=1.0,
                     color='magenta'))
            if not np.isfinite(aRTs_1) and np.isfinite(aRTs_2):
                ax.add_artist(patches.Rectangle(
                    ((gridpoints[i])[0], (gridpoints[i])[1]),
                     maxplot-(gridpoints[i])[0],
                     maxplot-(gridpoints[i])[1],
                     alpha=1.0,
                     color='orange'))
            continue # no finite aRT for >= 1 algo
        if not aRT_ratios[i] == 0:
            ax.add_artist(patches.Rectangle(
                ((gridpoints[i])[0], (gridpoints[i])[1]),
                 maxplot-(gridpoints[i])[0],
                 maxplot-(gridpoints[i])[1],
                 alpha=1.0,
                 color=aRTA_colormap(norm(aRT_ratios[i]))))
                
    
    cb = plt.colorbar(ticks=[-10, -5, 0, 5, 10])  # mainly to set correct tick values
    cb.ax.set_yticklabels(['10', '5', '0', '5', '10'])
    cb.set_label("<-- in favor of alg. B      aRTA ratio      in favor of alg. A -->")
    
    # beautify:
    ax.set_title("         aRTA ratio function plot for bbob-biobj function $f_{%d}$ (%d-D)" % (f_id, dim))
    [line.set_zorder(3) for line in ax.lines]
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
        filename = outputfolder + "aRT-ratios-f%02d-d%02d-%dx%dgrid" % (f_id, dim, n, n)
        if logscale:
            filename = filename + "-log"
        save_figure(filename)
    else:        
        plt.show(block=True)

    plt.close()

    
def compute_aRT(points, A):
    """
    Computes the average runtime to attain the objective vectors in points
    by the algorithm, with algorithm data given in dictionary A.
    
    Assumes that the algorithm data in A is given in the order of
    increasing number of function evaluations for each entry.
    
    >>> from cocopp.eaf import generate_aRTA_plot
    >>> A = {0: [[1, 1, 1], [3, 0.75, 0.5], [7, 0.5, 0.6]],
    ... 1: [[1, 0.9, 0.9], [2, 0.5, 0.4]]}
    >>> gridpoints = [[0.6, 0.5]]
    >>> generate_aRTA_plot.compute_aRT(gridpoints, A)
    array([ 9.])
    
    """
    
    
    sum_runtimes = np.zeros(len(points))
    num_runtimes_successful = np.zeros(len(points))
    
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
                        
        for i in range(len(points)):
            if runtime_to_attain_points[i] is np.nan:
                sum_runtimes[i] = sum_runtimes[i] + max_runtimes[i]
            else:
                sum_runtimes[i] = sum_runtimes[i] + runtime_to_attain_points[i]
                num_runtimes_successful[i] = num_runtimes_successful[i] + 1                

    aRT = np.zeros(len(points), dtype=float)

    for i in range(len(points)):
        if num_runtimes_successful[i] > 0:
            aRT[i] = sum_runtimes[i]/num_runtimes_successful[i]
        else:
            aRT[i] = np.nan

    return aRT
    
def weakly_dominates(a,b):
    """ Returns True iff a weakly dominates b wrt. minimization """
    
    return (a[0] <= b[0]) and (a[1] <= b[1])
    
def sample_down(B, n, logscale=True):
    """
        Samples down the data by only keeping one solution from B in each
        grid box (nxn grid within [0,maxplot] or [precision, maxplot] in the
        logscale case).
        
        The points, given in B (as [feval, f_1, f_2] vectors) are expected
        to be normalized such that ideal and nadir are [0,0] and [1,1]
        respectively.
        
    """
    
    C = np.array(B)
    C = C[C[:, 2].argsort(kind='mergesort')][::-1] # sort in descending order wrt second objective
    C = C[C[:, 1].argsort(kind='mergesort')][::-1] # now in descending order wrt first objective

    if logscale:
        # downsampling according to
        # np.logspace(np.log10(precision), np.log10(maxplot), num=n, endpoint=True, base=10.0)
        X = np.ceil((np.log10(C)-np.log10(precision))*(n-1)/(np.log10(maxplot)-np.log10(precision)))/((n-1)/(np.log10(maxplot)-np.log10(precision)))
    else:
        X = np.ceil(C*(n-1)/maxplot)/((n-1)/maxplot)

    # sort wrt second objective first
    idx_1 = X[:, 2].argsort(kind='mergesort')
    X = X[idx_1]
    # now wrt first objective to finally get a stable sort
    idx_2 = X[:, 1].argsort(kind='mergesort')
    X = X[idx_2]
    xflag = np.array([False] * len(X), dtype=bool)
    xflag[0] = True # always take the first point
    bestincell = 1
    for i in range(1, len(X)):
        if not (X[i, 1] == X[i-1, 1] and
                X[i, 2] == X[i-1, 2]):
            xflag[i] = True
            bestincell = i
        else:
            if X[i, 0] < X[bestincell, 0]:
                xflag[bestincell] = False
                xflag[i] = True
                bestincell = i
    X = ((C[idx_1])[idx_2])[xflag]
    B = X[X[:, 0].argsort(kind='mergesort')] # sort again wrt. #FEs

    return B

    
    
def DEPRECATED_sample_down(B, decimals):
    """ Samples down the solutions in B, given as (#funevals, f_1, f_2)
        entries in a list or an np.array such that only one of the solutions
        with the same objective vector is kept when they are rounded to the
        precision `10^{-decimal}`.
        
        >>> from cocopp.eaf import generate_aRTA_plot
        >>> A = [[1, 2.155, 3.342],
        ...      [2, 2.171, 3.326],
        ...      [2, 2.174, 3.329],
        ...      [4, 4, 2.2]]
        >>> generate_aRTA_plot.sample_down(A, 2)
        array([[ 1.   ,  2.155,  3.342],
               [ 2.   ,  2.171,  3.326],
               [ 4.   ,  4.   ,  2.2  ]])
        >>> generate_aRTA_plot.sample_down(A, 1)
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
    print('launching doctest on generate_aRTA_plot.py...')    
    t0 = time.time()
    import doctest
    doctest.testmod()
    print('** doctest finished in ', time.time() - t0, ' seconds')
