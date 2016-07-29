# -*- coding: utf-8 -*-
#
# Called by plots_alongDirections and doing the actual plotting.
#
# based on code by Thanh-Do Tran 2012--2015
# adapted by Dimo Brockhoff 2016

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

logscale = True
downsample = True
decimals=2 # precision for downsampling
maxplot = 2 # maximal displayed value (assuming nadir in [1,1])
precision = 5e-4 # smallest displayed value at maxplot*precision in logscale

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


def generate_ERD_plot(f_id, dim, inst_id, f1_id, f2_id, f1_instance, f2_instance,
                   outputfolder="./", inputfolder=None, tofile=True):
    ##############################################################
    #                                                            #
    # Objective Space plot indicating for each (grid) point      #
    # the runtime of the algorithm to attain it.                 #
    #                                                            #
    # Assumes that each instance is only contained once in the   #
    # data.                                                      #
    #                                                            #
    ##############################################################
    
    
    
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
                    if logscale:
                        newline[1] = np.log10((newline[1]-ideals[instance][0])/(nadirs[instance][0]-ideals[instance][0]))
                        newline[2] = np.log10((newline[2]-ideals[instance][1])/(nadirs[instance][1]-ideals[instance][1]))
                    else:
                        newline[1] = (newline[1]-ideals[instance][0])/(nadirs[instance][0]-ideals[instance][0])
                        newline[2] = (newline[2]-ideals[instance][1])/(nadirs[instance][1]-ideals[instance][1])

                    B.append(newline)
                    

            # store data of final instance:
            if instance not in A and not instance == -1:
                A[instance] = B
            print("all %d instances read in" % len(A))


    except:
        print("Problem opening %s" % (inputfolder + filename))
        e = sys.exc_info()[0]
        print("   Error: %s" % e)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # print all points of all runs
#    for key in A:
#        for a in A[key]:
#            plt.plot(a[1], a[2], 'xk')


#    for a in A[1]:
#        plt.plot(a[1], a[2], 'xk')
#    plt.show()

    
    # plot grid in normalized 2*[ideal, nadir]:
    n = 50 # number of grid points per objective
    if logscale:    
        #gridpoints = 10**(np.log10(maxplot) * (np.array(list(product(range(n),range(n))))/(n-1)))
        gridpoints = np.log10(maxplot * 10**(np.log10(precision)*np.array(list(product(range(n),range(n))))/(n-1)))
    else:
        gridpoints = maxplot * np.array(list(product(range(n),range(n))))/(n-1)
    
    colors = compute_aRT(gridpoints, A)

    # normalize colors:
    print(colors)
    colors = 255*(colors/max(colors))
    print(colors)
    
    #plt.scatter(gridpoints[:,0], gridpoints[:,1], c=colors, cmap='autumn_r', lw=0, norm=matplotlib.colors.LogNorm())
    plt.scatter(gridpoints[:,0], gridpoints[:,1], marker='s', c=colors, cmap='autumn_r', s=80, lw=0, norm=matplotlib.colors.LogNorm())
    
    

#    autumn = plt.get_cmap('autumn_r') 
#    scalarMap = matplotlib.cm.ScalarMappable(cmap=autumn)
#    scalarMap.set_norm(matplotlib.colors.LogNorm())

    
#    for i in range(len(gridpoints)):
#        if not np.isfinite(colors[i]):
#            continue # no finite aRT
#        if logscale:
#            ax.add_patch(patches.Rectangle(
#                ((gridpoints[i])[0], (gridpoints[i])[1]),
#                 np.log10(maxplot)-(gridpoints[i])[0],
#                 np.log10(maxplot)-(gridpoints[i])[1],
#                 alpha=1.0,
#                 color=matplotlib.cm.autumn_r(colors[i])),
#                 cmap='autumn_r')
#        else:
#            ax.add_patch(patches.Rectangle(
#                (gridpoints[:,0], gridpoints[:,1]), maxplot-gridpoints[:,0], maxplot-gridpoints[:,1],
#                alpha=0.05,
#                color='k'))
#            
    
    plt.colorbar()
    
    # beautify:
    ax.set_title("aRT in objective space for bbob-biobj function $f_{%d}$ (%d-D, %d instances)" % (f_id, dim, len(A)))
    [line.set_zorder(3) for line in ax.lines]
    [line.set_zorder(3) for line in ax.lines]
    #fig.subplots_adjust(left=0.1) # more room for the y-axis label
    if logscale:                
        ax.set_xlabel(r'log10($f_1 - f_1^\mathsf{opt}$) (normalized)', fontsize=16)
        ax.set_ylabel(r'log10($f_2 - f_2^\mathsf{opt}$) (normalized)', fontsize=16)    
        # we might want to zoom in a bit:
        ax.set_xlim((np.log10(precision), np.log10(maxplot)))
        ax.set_ylim((np.log10(precision), np.log10(maxplot)))
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
        saveFigure(filename, verbose=True)
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
#            print('..........................')
#            print(a)
            for i in range(len(points)):
                if not points_finished[i]:
                    if dominates(np.array([a[1], a[2]]), points[i]):
#                        print("(%e, %e) dominates (%e, %e)" % (a[1], a[2], points[i][0], points[i][1]))
#                        print("recorded runtime here: %d" % a[0])
                        runtime_to_attain_points[i] = a[0]
                        points_finished[i] = True
                    else:
#                        print("(%e, %e) does not dominate (%e, %e)" % (a[1], a[2], points[i][0], points[i][1]))
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
    
def dominates(a,b):
    """ returns True iff a dominates b wrt. minimization """
    
    return (a[0] < b[0]) and (a[1] < b[1])