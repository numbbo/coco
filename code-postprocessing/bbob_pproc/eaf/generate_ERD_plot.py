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
maxplot = 5 # maximal displayed value (assuming nadir in [1,1])
precision = 1e-3 # smallest displayed value in logscale
maxbudget = '1e6 * dim'  # largest budget for normalization of aRT-->sampling conversion
minbudget = '1'          # smallest budget for normalization of aRT-->sampling conversion
n = 50 # number of grid points per objective
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


def generate_ERD_plot(f_id, dim, f1_id, f2_id,
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

    fig = plt.figure(1)
    #fig.set_size_inches(fig.get_size_inches()[1], fig.get_size_inches()[1]) # make axes equal
    ax = fig.add_subplot(111)

    # print all (downsampled) points of all runs
#    for key in A:
#        for a in A[key]:
#            plt.plot(a[1], a[2], 'xk')


#    for a in A[1]:
#        plt.plot(a[1], a[2], 'ob')
    

    
    # plot grid in normalized [precision, maxplot]:
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

    colors = compute_aRT(gridpoints, A)

    # normalize colors:
    logcolors = np.log10(colors)
    logcolors = (logcolors - np.log10(eval(minbudget)))/(np.log10(eval(maxbudget))-np.log10(eval(minbudget)))

    # sort gridpoints (and of course colors) wrt. their aRT:
    idx = logcolors.argsort(kind='mergesort')
    #N = len(gridpoints)-1
    colors = colors[idx]
    logcolors = logcolors[idx]
    gridpoints = gridpoints[idx]

    if grayscale:
        erd_colormap = matplotlib.cm.gray_r
    else:
        erd_colormap = matplotlib.cm.hot_r

    for i in range(len(gridpoints)-1, -1, -1):
    #for i in range(1, len(gridpoints)-3, 1):
        if not np.isfinite(logcolors[i]):
            continue # no finite aRT
        ax.add_artist(patches.Rectangle(
                ((gridpoints[i])[0], (gridpoints[i])[1]),
                 maxplot-(gridpoints[i])[0],
                 maxplot-(gridpoints[i])[1],
                 alpha=1.0,
                 color=erd_colormap(logcolors[i])))
            
    #plt.scatter(gridpoints[:,0], gridpoints[:,1], marker='s', c=colors, cmap='autumn_r', s=80, lw=0, norm=matplotlib.colors.LogNorm())
    #plt.scatter(gridpoints[:,0], gridpoints[:,1], marker='s', c=colors, cmap='autumn_r', s=10, lw=0, norm=matplotlib.colors.LogNorm())    
        
    
    #plt.colorbar()
    
    # beautify:
    ax.set_title("aRT in objective space for bbob-biobj function $f_{%d}$ (%d-D, %d instances)" % (f_id, dim, len(A)))
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
    
    
def compute_aRT(points, A):
    """ Computes the average runtime to attain the objective vectors in points
        by the algorithm, with algorithm data given in dictionary A)
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
            if runtime_to_attain_points[i] == np.nan:
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