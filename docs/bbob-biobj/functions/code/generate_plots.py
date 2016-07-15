# -*- coding: utf-8 -*-
#
# Called by plots_alongDirections and doing the actual plotting.
#
# based on code by Thanh-Do Tran 2012--2015
# adapted by Dimo Brockhoff 2016

from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np  # "pip install numpy" installs numpy
import os
import sys

from bbob_pproc.ppfig import saveFigure

import bbobbenchmarks as bm
import paretofrontwrapper as pf # wrapper file and DLL must be in this folder


def generate_plots(f_id, dim, inst_id, f1_id, f2_id, f1_instance, f2_instance,
                   outputfolder="./", inputfolder=None, tofile=True, downsample=False):
    ##############################################################
    #                                                            #
    # Objective Space of points on cut (log-scale).              #
    #                                                            #
    ##############################################################
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    
    myc = ['g', 'b', 'r', 'y'] # colors for the different line directions
    myls = [':', '--', '-'] # line styles
    mylw = dict(lw=2, alpha=0.6) # line width # ALSO: mylw = {'lw':2, 'alpha':0.9}
    
    
    # define lines as a + t*b
    tlim = 10 # 
    ngrid = 10001
    t = np.linspace(-tlim, tlim, num=ngrid, endpoint=True)
    
    # Query the optimum from the benchmark to get a working objective function:
    # -------------------------------------
    f1, f1opt = bm.instantiate(f1_id, iinstance=f1_instance)
    f2, f2opt = bm.instantiate(f2_id, iinstance=f2_instance)
    
    fdummy = f1.evaluate(np.zeros((1, dim)))    
    xopt1 = f1.xopt # formerly: `f1.arrxopt[0]` but did not work for all functions
    f_xopt1 = [f1opt, f2.evaluate(xopt1)]
    
    fdummy = f2.evaluate(np.zeros((1, dim)))
    xopt2 = f2.xopt # formerly: `f2.arrxopt[0]` but did not work for all functions
    f_xopt2 = [f1.evaluate(xopt2), f2opt]
    
    nadir = np.array([f1.evaluate(xopt2), f2.evaluate(xopt1)])
    ideal = np.array([f1opt, f2opt])
    
    # evaluate points along random directions through single optima:
    #rand_dir_1 = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    rand_dir_1 = np.array([-2.57577836,  3.03082186, -1.33275642, -0.6939155 ,  0.99631351,
           -0.05842807,  1.99304198,  0.38531151,  1.3697517 ,  0.37240766,
            0.69762214, -0.79613309, -1.45320324, -0.97296174,  0.90871269,
           -1.00793426, -1.29250002,  0.25110439,  0.26014748, -0.1267351 ,
            0.63039621,  0.38236451,  1.07914151,  1.07130862,  0.13733215,
            1.97801217,  0.48601757,  2.3606844 ,  0.30784962, -0.36040267,
            0.68263725, -1.55353407, -0.57503424,  0.07362256,  0.95114969,
            0.43087735, -1.57600655,  0.48304268, -0.88184912,  1.85066177])[0:dim]
    rand_dir_1 = rand_dir_1/np.linalg.norm(rand_dir_1)
    #rand_dir_2 = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    rand_dir_2 = np.array([0.2493309 , -2.05353785, -1.08038135, -0.06152298, -0.37996052,
           -0.65976313, -0.11217795, -1.41055602,  0.20321651, -1.42727459,
           -0.09742259, -0.26135753, -0.20899801,  0.85056449, -0.58492263,
           -0.93028813, -0.6576416 , -0.02854442, -0.53294699, -0.40898327,
           -0.64209791,  0.62349299, -0.44248805,  0.60715229,  0.97420653,
           -0.40989115,  0.67065727,  0.23491168, -0.0607614 , -0.42400703,
           -1.77536414,  1.92731362,  2.38098092, -0.23789751, -0.02411066,
           -0.37445709,  0.43547281,  0.32148583, -0.4257802 ,  0.15550121])[0:dim]
    rand_dir_2 = rand_dir_2/np.linalg.norm(rand_dir_2)
    rand_dir_3 = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
#    rand_dir_3 = np.array([0.27274996,  0.09450028,  0.23123471, -0.17268026, -0.19352246,
#            0.11116155,  1.91171592, -0.77188094,  0.50033182, -2.93726319,
#           -0.0444466 , -0.83483599, -1.05971685,  0.35220208,  0.67446614,
#           -0.66144976,  0.15873096,  0.63002013, -0.75455445,  0.11553671,
#            0.53268058, -0.17107212, -2.68158842,  1.76162118, -1.10528215,
#           -1.3174873 , -0.56827552,  0.8938743 , -1.40129273,  1.24724136,
#            0.32995442,  1.64754152, -0.23038488, -0.1996612 ,  0.7423728 ,
#            0.41590582, -0.49735973, -0.16317831,  0.14116915,  0.33144299])[0:dim]
#    rand_dir_3 = rand_dir_3/np.linalg.norm(rand_dir_3)    
    rand_dir_4 = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
#    rand_dir_4 = np.array([-1.64810074,  0.06035188, -1.08343971,  0.69871916, -1.57870908,
#            -0.39555544,  1.15952858,  0.82573846, -1.00821565,  0.46347426,
#            0.46817715, -0.70617468, -0.56754204, -1.77903594, -0.15184591,
#            2.10968445,  0.53652335, -0.03221351, -0.34664564,  1.69246492,
#            1.26043695,  0.20284844,  1.90425762, -0.43203046,  0.33297092,
#           -0.43151518, -0.27561938, -0.64456918, -1.52515793,  0.16840333,
#           -1.44740417, -0.07328904, -0.74026773,  0.02869038, -0.65416703,
#            0.55212071, -1.13507935, -1.18781606,  0.42888208, -1.47626463])[0:dim]
    rand_dir_4 = rand_dir_4/np.linalg.norm(rand_dir_4)
        
    
    # now sample two random points
    # rand_x_1 = -4+8*np.random.rand(dim)
    rand_x_1 = np.array([-2.70496645, -0.39106794, -2.80086174, -3.66756864,  2.14644397,
        2.78153367,  1.56329668,  2.35839362,  0.13302063, -2.91032329,
       -2.51556623, -2.35077186,  2.58377453,  1.17508714, -2.4457919 ,
        1.45033066, -1.23112017, -2.25318184,  2.41933833, -1.14164988,
       -2.36275527, -3.25853312, -2.4609917 ,  3.48296483, -2.68189074,
       -2.05345914, -2.4116529 ,  3.08138791, -2.23247829,  2.54796847,
       -0.936912  ,  3.35564688,  0.51737322, -0.92592536,  1.65481046,
       -2.52985307,  3.7431933 , -3.6630677 , -0.40448911,  1.33128767])[0:dim]    
    # rand_x_2 = -4+8*np.random.rand(dim)
    rand_x_2 = np.array([1.57461786, -3.44804825, -3.81020969,  2.83971589,  3.27253056,
       -3.26623201,  3.79526151,  1.76316424,  1.79345621, -0.81215354,
        2.06356913,  1.02657347,  2.99781081,  0.35872047,  3.69835244,
       -1.68708122,  1.84948801, -0.86589091, -1.61500454, -1.03210602,
        3.96363037, -1.30389274,  2.16486049, -2.77809263, -2.78117177,
       -0.89747482,  3.85189385,  2.34298403,  1.45079637,  3.78130948,
        2.55578938,  2.23402556,  0.79451819,  0.30563072,  1.91404655,
        0.37739932, -2.07692776, -0.06961333, -2.73583526, -2.70524468])[0:dim]    
    
    # Construct solutions along rand_dir_1 through xopt1
    # ------------------------------------------------------
    xgrid_opt_1 = np.tile(xopt1, (ngrid, 1))
    xgrid_opt_1 = np.array(xgrid_opt_1 + np.dot(t.reshape(ngrid,1), np.array([rand_dir_1])))
    
    # Construct solutions along coordinate axes through xopt1
    # -------------------------------------------------------
    xgrid_opt_1_along_axes = []
    for k in range(dim):
        xgrid_along_axis = np.tile(xopt1, (ngrid, 1))
        x_dir = np.zeros(dim)
        x_dir[k] = 1
        xgrid_along_axis = xgrid_along_axis + np.dot(t.reshape(ngrid,1), np.array([x_dir]))
        xgrid_opt_1_along_axes.append(xgrid_along_axis)
    xgrid_opt_1_along_axes = np.array(xgrid_opt_1_along_axes)
    
    # Construct solutions along rand_dir_2 through xopt2
    # ------------------------------------------------------
    xgrid_opt_2 = np.tile(xopt2, (ngrid, 1))
    xgrid_opt_2 = np.array(xgrid_opt_2 + np.dot(t.reshape(ngrid,1), np.array([rand_dir_2])))
    
    # Construct solutions along coordinate axes through xopt1
    # -------------------------------------------------------
    xgrid_opt_2_along_axes = []
    for k in range(dim):
        xgrid_along_axis = np.tile(xopt2, (ngrid, 1))
        x_dir = np.zeros(dim)
        x_dir[k] = 1
        xgrid_along_axis = xgrid_along_axis + np.dot(t.reshape(ngrid,1), np.array([x_dir]))
        xgrid_opt_2_along_axes.append(xgrid_along_axis)
    xgrid_opt_2_along_axes = np.array(xgrid_opt_2_along_axes)
        
    
    # Construct solutions along line through xopt1 and xopt2
    # ------------------------------------------------------
    xgrid_12 = np.tile((xopt1+xopt2)/2, (ngrid, 1))
    xgrid_12 = np.array(xgrid_12 + np.dot(t.reshape(ngrid,1),
                        np.array([xopt2-xopt1])/np.linalg.norm([xopt2-xopt1])
                        )
               )
               
    # Construct solutions along a fully random line
    # ------------------------------------------------------
    xgrid_rand_1 = np.tile(rand_x_1, (ngrid, 1))
    xgrid_rand_1 = np.array(xgrid_rand_1
                   + np.dot(t.reshape(ngrid,1), np.array([rand_dir_3])))

    # and for another fully random line
    # ------------------------------------------------------
    xgrid_rand_2 = np.tile(rand_x_2, (ngrid, 1))
    xgrid_rand_2 = np.array(xgrid_rand_2
                   + np.dot(t.reshape(ngrid,1), np.array([rand_dir_4])))
    
    
    # Evaluate the grid for each direction
    # -------------------------------------------
    fgrid_opt_1 = [f1.evaluate(xgrid_opt_1), f2.evaluate(xgrid_opt_1)]
    fgrid_opt_2 = [f1.evaluate(xgrid_opt_2), f2.evaluate(xgrid_opt_2)]
    fgrid_12 = [f1.evaluate(xgrid_12), f2.evaluate(xgrid_12)]
    fgrid_rand_1 = [f1.evaluate(xgrid_rand_1), f2.evaluate(xgrid_rand_1)]
    fgrid_rand_2 = [f1.evaluate(xgrid_rand_2), f2.evaluate(xgrid_rand_2)]
    fgrid_opt_1_along_axes = []
    for k in range(dim):    
        fgrid_opt_1_along_axes.append([f1.evaluate(xgrid_opt_1_along_axes[k]),
                                       f2.evaluate(xgrid_opt_1_along_axes[k])])                               
    fgrid_opt_2_along_axes = []
    for k in range(dim):    
        fgrid_opt_2_along_axes.append([f1.evaluate(xgrid_opt_2_along_axes[k]),
                                       f2.evaluate(xgrid_opt_2_along_axes[k])])                               
    
    # plot reference sets if available:
    if inputfolder:
        filename = "bbob-biobj_f%02d_i%02d_d%02d_nondominated.adat" % (f_id, inst_id, dim)
        try:
            A = np.array(np.loadtxt(inputfolder + filename, comments='%', usecols = (1,2)))
        except:
            print("Problem opening %s" % (inputfolder + filename))
            e = sys.exc_info()[0]
            print("   Error: %s" % e)

        
        if downsample:
            # normalize A wrt ideal and nadir (and take care of having no inf
            # in data by adding the constant 1e-15 before the log10):
            B = (A-ideal) / (nadir-ideal)
            Blog = np.log10((A-ideal) / (nadir-ideal) + 1e-15)
            # cut precision to downsample:
            decimals=3
            B = np.around(B, decimals=decimals)
            Blog = np.around(Blog, decimals=decimals)
            
            if 11<3: # filter out dominated points (and doubles)
                pfFlag = pf.callParetoFront(B)
                pfFlaglog = pf.callParetoFront(Blog)
            else: # filter out all but one point per grid cell
                pfFlag = np.array([False] * len(B), dtype=bool)
                # check corner case first:
                if not (B[2][0] == B[0][0] and B[2][1] == B[0][1]):
                    pfFlag[2] = True
                else:
                    B[2] = B[0]
                for i in range(3,len(B)):
                    if not (B[i][0] == B[i-1][0] and B[i][1] == B[i-1][1]):
                        pfFlag[i] = True

                pfFlaglog = np.array([False] * len(Blog), dtype=bool)
                # check corner case first:
                if not (Blog[2][0] == Blog[0][0] and Blog[2][1] == Blog[0][1]):
                    pfFlaglog[2] = True
                else:
                    Blog[2] = Blog[0]
                for i in range(3,len(Blog)):
                    if not (Blog[i][0] == Blog[i-1][0] and Blog[i][1] == Blog[i-1][1]):
                        pfFlaglog[i] = True

            # ensure that both extremes are still in, assuming they are stored in the beginning:
            pfFlag[0] = True
            pfFlaglog[0] = True
            pfFlag[1] = True
            pfFlaglog[1] = True            
            Alog = A[pfFlaglog]
            A = A[pfFlag]
            # finally sort wrt f_1 axis:
            Alog = Alog[Alog[:,0].argsort(kind='mergesort')]
            A = A[A[:,0].argsort(kind='mergesort')]
            

        # normalized plot, such that ideal and nadir are mapped to
        # 0 and 1 respectively; add 1e-15 for numerical reasons (to not have
        # inf in the data to plot)
        plt.loglog((Alog[:,0] - ideal[0])/(nadir[0]-ideal[0]) + 1e-15,
                   (Alog[:,1] - ideal[1])/(nadir[1]-ideal[1]) + 1e-15,
                   '.k', markersize=8)
        
    # plot actual solutions along directions:
    numticks = 5
    nf = nadir-ideal # normalization factor used very often now
    for k in range(dim):    
        p6, = ax.loglog(((fgrid_opt_1_along_axes[k])[0]-f1opt)/nf[0],
                        ((fgrid_opt_1_along_axes[k])[1]-f2opt)/nf[1],
                        color=myc[1], ls=myls[0], lw=1, alpha=0.3)
    for k in range(dim):    
        p7, = ax.loglog(((fgrid_opt_2_along_axes[k])[0]-f1opt)/nf[0],
                        ((fgrid_opt_2_along_axes[k])[1]-f2opt)/nf[1],
                        color=myc[1], ls=myls[0], lw=1, alpha=0.3)
            
    p1, = ax.loglog((fgrid_opt_1[0]-f1opt)/nf[0], (fgrid_opt_1[1]-f2opt)/nf[1], color=myc[1], ls=myls[2],
                    label=r'cuts through single optima', **mylw)
    p2, = ax.loglog((fgrid_opt_2[0]-f1opt)/nf[0], (fgrid_opt_2[1]-f2opt)/nf[1], color=myc[1], ls=myls[2],
                    **mylw)
    p3, = ax.loglog((fgrid_12[0]-f1opt)/nf[0], (fgrid_12[1]-f2opt)/nf[1],
                    color=myc[2], ls=myls[2],
                    label=r'cut through both optima', **mylw)
    p4, = ax.loglog((fgrid_rand_1[0]-f1opt)/nf[0], (fgrid_rand_1[1]-f2opt)/nf[1],
                    color=myc[3], ls=myls[2],
                    label=r'two random directions', **mylw)
    p5, = ax.loglog((fgrid_rand_2[0]-f1opt)/nf[0], (fgrid_rand_2[1]-f2opt)/nf[1],
                    color=myc[3], ls=myls[2], **mylw)
        
    

    # print 'ticks' along the axes in equidistant t space:
    numticks = 11
    plot_ticks([fgrid_opt_1[0], fgrid_opt_1[1]], numticks, nadir, ideal, ax, mylw, myc[1], logscale=True)
    plot_ticks([fgrid_opt_2[0], fgrid_opt_2[1]], numticks, nadir, ideal, ax, mylw, myc[1], logscale=True)
    plot_ticks([fgrid_12[0], fgrid_12[1]], numticks, nadir, ideal, ax, mylw, myc[2], logscale=True)
    plot_ticks([fgrid_rand_1[0], fgrid_rand_1[1]], numticks, nadir, ideal, ax, mylw, myc[3], logscale=True)
    plot_ticks([fgrid_rand_2[0], fgrid_rand_2[1]], numticks, nadir, ideal, ax, mylw, myc[3], logscale=True)
    
    # Get Pareto front from vectors of objective values obtained
    objs = np.vstack((fgrid_opt_1[0], fgrid_opt_1[1])).transpose()
    pfFlag_opt_1 = pf.callParetoFront(objs)
    ax.loglog((fgrid_opt_1[0][pfFlag_opt_1]-f1opt)/nf[0],
              (fgrid_opt_1[1][pfFlag_opt_1]-f2opt)/nf[1],
              color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
              alpha=0.4)
    objs = np.vstack((fgrid_opt_2[0], fgrid_opt_2[1])).transpose()
    pfFlag_opt_2 = pf.callParetoFront(objs)
    ax.loglog((fgrid_opt_2[0][pfFlag_opt_2]-f1opt)/nf[0],
              (fgrid_opt_2[1][pfFlag_opt_2]-f2opt)/nf[1],
              color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
              alpha=0.4)
    objs = np.vstack((fgrid_12[0], fgrid_12[1])).transpose()
    pfFlag_12 = pf.callParetoFront(objs)
    ax.loglog((fgrid_12[0][pfFlag_12]-f1opt)/nf[0],
              (fgrid_12[1][pfFlag_12]-f2opt)/nf[1],
              color=myc[2], ls='', marker='.', markersize=8, markeredgewidth=0,
              alpha=0.4)
    objs = np.vstack((fgrid_rand_1[0], fgrid_rand_1[1])).transpose()
    pfFlag_rand_1 = pf.callParetoFront(objs)
    ax.loglog((fgrid_rand_1[0][pfFlag_rand_1]-f1opt)/nf[0],
              (fgrid_rand_1[1][pfFlag_rand_1]-f2opt)/nf[1],
              color=myc[3], ls='', marker='.', markersize=8, markeredgewidth=0,
              alpha=0.4)
    objs = np.vstack((fgrid_rand_2[0], fgrid_rand_2[1])).transpose()
    pfFlag_rand_2 = pf.callParetoFront(objs)
    ax.loglog((fgrid_rand_2[0][pfFlag_rand_2]-f1opt)/nf[0],
              (fgrid_rand_2[1][pfFlag_rand_2]-f2opt)/nf[1],
              color=myc[3], ls='', marker='.', markersize=8, markeredgewidth=0,
              alpha=0.4)
    
    
    # plot nadir:
    ax.loglog((nadir[0]-f1opt)/nf[0], (nadir[1]-f2opt)/nf[1],
              color='k', ls='', marker='+', markersize=9, markeredgewidth=1.5,
              alpha=0.9)
    
    
    # beautify:
    ax.set_xlabel(r'$f_1 - f_1^\mathsf{opt}$ (normalized)', fontsize=16)
    ax.set_ylabel(r'$f_2 - f_2^\mathsf{opt}$ (normalized)', fontsize=16)
    ax.legend(loc="best", framealpha=0.2)
    ax.set_title("bbob-biobj $f_{%d}$ along linear search space directions (%d-D, instance %d)" % (f_id, dim, inst_id))
    [line.set_zorder(3) for line in ax.lines]
    [line.set_zorder(3) for line in ax.lines]
    fig.subplots_adjust(left=0.1) # more room for the y-axis label
    
    # we might want to zoom in a bit:
    ax.set_xlim((1e-3, plt.xlim()[1]))
    ax.set_ylim((1e-3, plt.ylim()[1]))
    #    ax.set_ylim((0, 2*(nadir[1] - f2opt)))
    
    # add rectangle as ROI
    ax.add_patch(patches.Rectangle(
            ((ideal[0]-f1opt)/nf[0] + 1e-16, (ideal[1]-f2opt)/nf[1] + 1e-16),
             (nadir[0]-ideal[0])/nf[0], (nadir[1]-ideal[1])/nf[1],
             alpha=0.05,
             color='k'))
    
    if tofile:
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        filename = outputfolder + "directions-f%02d-i%02d-d%02d-logobjspace" % (f_id, inst_id, dim)
        saveFigure(filename, verbose=True)
    else:   
        plt.show(block=True)
        
    plt.close()    
    
    
    
    
    
    ##############################################################
    #                                                            #
    # Plot the same, but not in log-scale.                       #
    #                                                            #
    ##############################################################
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    
    # plot reference sets if available:
    if inputfolder:
        plt.plot(A[:,0], A[:,1], '.k', markersize=8)
    
    
    for k in range(dim):    
        p6, = ax.plot((fgrid_opt_1_along_axes[k])[0],
                      (fgrid_opt_1_along_axes[k])[1],
                      color=myc[1], ls=myls[0], lw=1, alpha=0.3)
    for k in range(dim):    
        p7, = ax.plot((fgrid_opt_2_along_axes[k])[0],
                      (fgrid_opt_2_along_axes[k])[1],
                      color=myc[1], ls=myls[0], lw=1, alpha=0.3)    
    p1, = ax.plot(fgrid_opt_1[0], fgrid_opt_1[1], color=myc[1], ls=myls[2],
                    label=r'cuts through single optima', **mylw)
    
    p2, = ax.plot(fgrid_opt_2[0], fgrid_opt_2[1], color=myc[1], ls=myls[2],
                    **mylw)
    
    p3, = ax.plot(fgrid_12[0], fgrid_12[1], color=myc[2], ls=myls[2],
                    label=r'cut through both optima', **mylw)
    
    p4, = ax.plot(fgrid_rand_1[0], fgrid_rand_1[1], color=myc[3], ls=myls[2],
                    label=r'two random directions', **mylw)
    
    p4, = ax.plot(fgrid_rand_2[0], fgrid_rand_2[1], color=myc[3], ls=myls[2],
                    **mylw)
        
    # plot a few ticks along directions, equi-distant in search space:
    numticks = 11
    plot_ticks(fgrid_opt_1, numticks, nadir, ideal, ax, mylw, 'b')
    plot_ticks(fgrid_opt_2, numticks, nadir, ideal, ax, mylw, 'b')
    plot_ticks(fgrid_12, numticks, nadir, ideal, ax, mylw, 'r')
    plot_ticks(fgrid_rand_1, numticks, nadir, ideal, ax, mylw, 'y')
    plot_ticks(fgrid_rand_2, numticks, nadir, ideal, ax, mylw, 'y')

    
    
    # plot non-dominated points
    ax.plot(fgrid_opt_1[0][pfFlag_opt_1], fgrid_opt_1[1][pfFlag_opt_1], color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(fgrid_opt_2[0][pfFlag_opt_2], fgrid_opt_2[1][pfFlag_opt_2], color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(fgrid_12[0][pfFlag_12], fgrid_12[1][pfFlag_12], color=myc[2], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(fgrid_rand_1[0][pfFlag_rand_1], fgrid_rand_1[1][pfFlag_rand_1], color=myc[3], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(fgrid_rand_2[0][pfFlag_rand_2], fgrid_rand_2[1][pfFlag_rand_2], color=myc[3], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
        

    # plot nadir:
    ax.plot(nadir[0], nadir[1], color='k', ls='', marker='+', markersize=9, markeredgewidth=1.5,
                                 alpha=0.9)
    # plot ideal:
    ax.plot(ideal[0], ideal[1], color='k', ls='', marker='x', markersize=8, markeredgewidth=1.5,
                                 alpha=0.9)

    # plot extremes    
    ax.plot(f_xopt1[0], f_xopt1[1], color='blue', ls='', marker='*', markersize=8, markeredgewidth=0.5, markeredgecolor='black')
    ax.plot(f_xopt2[0], f_xopt2[1], color='blue', ls='', marker='*', markersize=8, markeredgewidth=0.5, markeredgecolor='black')
    
    
    # beautify:
    ax.set_xlabel(r'first objective', fontsize=16)
    ax.set_ylabel(r'second objective', fontsize=16)
    ax.legend(loc="best", framealpha=0.2)
    ax.set_title("bbob-biobj $f_{%d}$ along linear search space directions (%d-D, instance %d)" % (f_id, dim, inst_id))    
    [line.set_zorder(3) for line in ax.lines]
    [line.set_zorder(3) for line in ax.lines]
    fig.subplots_adjust(left=0.1) # more room for the y-axis label
    
    # zoom into Pareto front:
    ax.set_xlim((ideal[0]-0.05*(nadir[0] - ideal[0]), nadir[0] + (nadir[0] - ideal[0])))
    ax.set_ylim([ideal[1]-0.05*(nadir[1] - ideal[1]), nadir[1] + (nadir[1] - ideal[1])])
    
    # add rectangle as ROI
    ax.add_patch(patches.Rectangle(
            (ideal[0], ideal[1]), nadir[0]-ideal[0], nadir[1]-ideal[1],
            alpha=0.05,
            color='k'))
    
    if tofile:
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        filename = outputfolder + "directions-f%02d-i%02d-d%02d-objspace" % (f_id, inst_id, dim)
        saveFigure(filename, verbose=True)
    else:        
        plt.show(block=True)
    
    plt.close()
    
    
    ##############################################################
    #                                                            #
    # Finally, the corresponding plots in search space, i.e.     #
    # projections of it onto the variables x_1 and x_(dim-1)     #
    # (or x1, x2 in the case of not enough variables).           #
    #                                                            #
    ##############################################################
    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    
    # plot reference sets if available:
    #if inputfolder:
    #    plt.plot(A[:,0], A[:,1], '.k', markersize=8)
    
    ax.set_xlabel(r'$x_1$', fontsize=16)
    # fix second variable in addition to x_1:
    if dim > 2:
        second_variable = -2
        ax.set_ylabel(r'$x_{%d}$' % (dim-1), fontsize=16)
    else:
        second_variable = 1
        ax.set_ylabel(r'$x_{%d}$' % dim, fontsize=16)
    
    # read and plot best Pareto set approximation
    if inputfolder:
        filename = "bbob-biobj_f%02d_i%02d_d%02d_nondominated.adat" % (f_id, inst_id, dim)
        C = []
        with open(inputfolder + filename) as f:
            for line in f:
                splitline = line.split()
                if len(splitline) == (dim + 3):  # has line x-values?
                    C.append(np.array(splitline[3:], dtype=np.float))
        C = np.array(C)
        C = C[C[:, second_variable].argsort(kind='mergesort')] # sort wrt x_{second_variable} first
        C = C[C[:, 0].argsort(kind='mergesort')] # now wrt x_1 to finally get a stable sort
        pareto_set_approx_size = C.shape[0]

        # filter out all but one point per grid cell in the 
        # (x_1, x_{second_variable}) space
        if downsample:
            decimals=2
            X = np.around(C, decimals=decimals)
            # sort wrt x_{second_variable} first
            idx_1 = X[:, second_variable].argsort(kind='mergesort')
            X = X[idx_1] 
            # now wrt x_1 to finally get a stable sort
            idx_2 = X[:, 0].argsort(kind='mergesort')
            X = X[idx_2]
            xflag = np.array([False] * len(X), dtype=bool)
            xflag[0] = True # always take the first point
            for i in range(1, len(X)):
                if not (X[i,0] == X[i-1,0] and
                        X[i,second_variable] == X[i-1, second_variable]):
                    xflag[i] = True
            X = ((C[idx_1])[idx_2])[xflag]

        pareto_set_sample_size = X.shape[0]
        
        paretosetlabel = ('reference set (%d of %d points)' %
                          (pareto_set_sample_size, pareto_set_approx_size))
        plt.plot(X[:, 0], X[:, second_variable], '.k', markersize=8,
                 label=paretosetlabel)
    # end of reading in and plotting best Pareto set approximation

    for k in range(dim):    
        p6, = ax.plot(xgrid_opt_1_along_axes[k][:, 0],
                      xgrid_opt_1_along_axes[k][:, second_variable],
                      color=myc[1], ls=myls[0], lw=1, alpha=0.3)
    for k in range(dim):
        p7, = ax.plot(xgrid_opt_2_along_axes[k][:, 0],
                      xgrid_opt_2_along_axes[k][:, second_variable],
                      color=myc[1], ls=myls[0], lw=1, alpha=0.3)

    p1, = ax.plot(xgrid_opt_1[:, 0], xgrid_opt_1[:, second_variable], color=myc[1], ls=myls[2],
                    label=r'cuts through single optima', **mylw)

    p2, = ax.plot(xgrid_opt_2[:, 0], xgrid_opt_2[:, second_variable], color=myc[1], ls=myls[2],
                    **mylw)

    p3, = ax.plot(xgrid_12[:, 0], xgrid_12[:, second_variable], color=myc[2], ls=myls[2],
                    label=r'cut through both optima', **mylw)

    p4, = ax.plot(xgrid_rand_1[:, 0], xgrid_rand_1[:, second_variable], color=myc[3], ls=myls[2],
                    label=r'two random directions', **mylw)

    p5, = ax.plot(xgrid_rand_2[:, 0], xgrid_rand_2[:, second_variable], color=myc[3], ls=myls[2],
                    **mylw)

    # plot non-dominated points
    ax.plot(xgrid_opt_1[pfFlag_opt_1, 0], xgrid_opt_1[pfFlag_opt_1, second_variable], color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(xgrid_opt_2[pfFlag_opt_2, 0], xgrid_opt_2[pfFlag_opt_2, second_variable], color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(xgrid_12[pfFlag_12, 0], xgrid_12[pfFlag_12, second_variable], color=myc[2], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(xgrid_rand_1[pfFlag_rand_1, 0], xgrid_rand_1[pfFlag_rand_1, second_variable], color=myc[3], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)
    ax.plot(xgrid_rand_2[pfFlag_rand_2, 0], xgrid_rand_2[pfFlag_rand_2, second_variable], color=myc[3], ls='', marker='.', markersize=8, markeredgewidth=0,
                                 alpha=0.4)


    # highlight the region [-5,5]
    ax.add_patch(patches.Rectangle(
            (-5, -5), 10, 10,
            alpha=0.05,
            color='k'))
    
    # beautify
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    if dim == 2:
        ax.set_title("decision space of bbob-biobj $f_{%d}$ (%d-D, instance %d)" % (f_id, dim, inst_id))    
    else:
        ax.set_title("projection of decision space for bbob-biobj $f_{%d}$ (%d-D, instance %d)" % (f_id, dim, inst_id))    
    ax.legend(loc="best", framealpha=0.2, numpoints=1)
    fig.subplots_adjust(left=0.1) # more room for the y-axis label    
    
    # printing
    if tofile:
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        filename = outputfolder + "directions-f%02d-i%02d-d%02d-searchspace" % (f_id, inst_id, dim)
        saveFigure(filename, verbose=True)
    else:        
        plt.show(block=True)
    
    plt.close()
    
    
    
    
    
def plot_ticks(linepoints, numticks, nadir, ideal, ax, mylw, myc, logscale=False):
    """
    Plots numticks equi-distant ticks perpendicular to the line defined by
    linepoints. nadir gives the nadir point and ideal the ideal point for
    normalizing the ticks' length, ax gives the subplot to plot in and 
    mylw and myc the line width and color parameters respectively.
    
    logscale==True enables plotting for loglog scale plots of objective space
    """

    # normalization in case of logscale:
    if logscale:
        linepoints[0] = (np.array(linepoints[0])-ideal[0])/(nadir[0]-ideal[0])
        linepoints[1] = (np.array(linepoints[1])-ideal[1])/(nadir[1]-ideal[1])              
    
    t_idx = np.linspace(0, len(linepoints[0])-1, num=numticks, endpoint=True, dtype=int)
    for i in range(1,numticks-1): # don't put ticks on extremes of the lines
        before = np.array([linepoints[0][t_idx[i]-1], linepoints[1][t_idx[i]-1]])
        after = np.array([linepoints[0][t_idx[i]+1], linepoints[1][t_idx[i]+1]])                            
  
        if logscale:
            # rotation by 90 degrees to get direction in log-scale
            if abs((after-before)[0]) < 1e-15:
                # handle case by hand where log might not work:
                if abs((after-before)[1]) + (after-before)[1] > 0:
                    # rotated direction is negative
                    direction = np.array([-0.1,0])
                else:
                    direction = np.array([0.1, 0])
            elif abs((after-before)[1]) < 1e-15:
                # handle the opposite case by hand where log might not work:
                if abs((after-before)[0]) + (after-before)[0] > 0:
                    # rotated direction is positive
                    direction = np.array([0, 0.1])
                else:
                    direction = np.array([0, -0.1])
            else:
                direction = np.dot(np.log10(after)-np.log10(before), np.array([[0, -1], [1, 0]]))
                # normalize length
                direction = 0.1 * direction / np.linalg.norm(direction)

            if (linepoints[0][t_idx[i]] <=0) or (linepoints[1][t_idx[i]] <=0):
                continue
            
            startpoint = [np.log10(linepoints[0][t_idx[i]]) + direction[0]*0.2, np.log10(linepoints[0][t_idx[i]]) + direction[0]*1.1]
            endpoint = [np.log10(linepoints[1][t_idx[i]]) + direction[1]*0.2, np.log10(linepoints[1][t_idx[i]]) + direction[1]*1.1]
            
            
            # transform everything back to original space before loglog plot
            startpoint[0] = 10**startpoint[0]
            startpoint[1] = 10**startpoint[1]
            endpoint[0] = 10**endpoint[0]
            endpoint[1] = 10**endpoint[1]
            
            ax.loglog(startpoint, endpoint,
                      color=myc, ls='-', **mylw)
                      
            
        else:
            # rotation by 90 degrees to get direction
            direction = np.dot((after-before)/(nadir-ideal), np.array([[0, -1], [1, 0]]))
            # normalize length and scale again to full objective space
            direction = 0.02 * direction / np.linalg.norm(direction)
            direction = direction * (nadir-ideal)
            ax.plot([linepoints[0][t_idx[i]] + direction[0]*0.2, linepoints[0][t_idx[i]] + direction[0]*1.1],
                    [linepoints[1][t_idx[i]] + direction[1]*0.2, linepoints[1][t_idx[i]] + direction[1]*1.1],
                    color=myc, ls='-', **mylw)
    return ax