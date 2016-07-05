#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# based on code by Thanh-Do Tran 2012--2015
# adapted by Dimo Brockhoff 2016

from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange  # let range always be an iterator
except NameError: pass
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
verbose = 1

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, patches
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

import bbobbenchmarks as bm
import paretofrontwrapper as pf # wrapper file and DLL must be in this folder




suite_name = "bbob-biobj"
suite_instance = "year:2016"
suite_options = "dimensions: 2,3,5,10,20,40"

suite = Suite(suite_name, suite_instance, suite_options)






f1_id = 8 # in function_ids
f2_id = 15
f1_instance = 2 # in instances # 0 seems to be the instance used in bbobdocfunctions.pdf
f2_instance = 3
dim = 5 # in dimensions








##############################################################
#                                                            #
# Objective Space of points on cut (log-scale).              #
#                                                            #
##############################################################

fig = plt.figure(2)
ax = fig.add_subplot(111)

myc = ['g', 'b', 'r'] # colors for the different line directions
myls = [':', '--', '-'] # line styles
mydash = ([1, 5, 1, 5], [10, 5, 10, 5], [10, 2, 10, 2]) # format: line length, space length, ...
mydashcap = ['round', 'butt', 'butt']
mylw = dict(lw=2, alpha=0.6) # line width # ALSO: mylw = {'lw':2, 'alpha':0.9}
mytkl = dict(size=4, width=1.2)


# define lines as a + t*b
tlim = 10 # 
ngrid = 10000
t = np.linspace(-tlim, tlim, num=ngrid, endpoint=True)

# Query the optimum from the benchmark to get a working objective function:
# -------------------------------------
f1, f1opt = bm.instantiate(f1_id, iinstance=f1_instance)
f2, f2opt = bm.instantiate(f2_id, iinstance=f2_instance)

fdummy = f1.evaluate(np.zeros((1, dim)))
xopt1 = f1.arrxopt[0] # 0 as shape(f.arrxopt) == xpop.shape
f_xopt1 = [f1opt, f2.evaluate(xopt1)]

fdummy = f2.evaluate(np.zeros((1, dim)))
xopt2 = f2.arrxopt[0]
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


# Construct solutions along rand_dir_1 through xopt1
# ------------------------------------------------------
xgrid_rand_1 = np.tile(xopt1, (ngrid, 1))
xgrid_rand_1 = xgrid_rand_1 + np.dot(t.reshape(ngrid,1), np.array([rand_dir_1]))

# Construct solutions along rand_dir_2 through xopt2
# ------------------------------------------------------
xgrid_rand_2 = np.tile(xopt2, (ngrid, 1))
xgrid_rand_2 = xgrid_rand_2 + np.dot(t.reshape(ngrid,1), np.array([rand_dir_2]))

# Construct solutions along line through xopt1 and xopt2
# ------------------------------------------------------
xgrid_12 = np.tile(xopt1, (ngrid, 1))
xgrid_12 = xgrid_12 + np.dot(t.reshape(ngrid,1), np.array([xopt2-xopt1]))

# Evaluate the grid for each direction
# -------------------------------------------
fgrid_rand_1 = [f1.evaluate(xgrid_rand_1), f2.evaluate(xgrid_rand_1)]
fgrid_rand_2 = [f1.evaluate(xgrid_rand_2), f2.evaluate(xgrid_rand_2)]
fgrid_12 = [f1.evaluate(xgrid_12), f2.evaluate(xgrid_12)]

p1, = ax.loglog(fgrid_rand_1[0]-f1opt, fgrid_rand_1[1]-f2opt, color=myc[0], ls=myls[2],
                label=r'through $\mathsf{opt}_1$', **mylw)

p2, = ax.loglog(fgrid_rand_2[0]-f1opt, fgrid_rand_2[1]-f2opt, color=myc[1], ls=myls[2],
                label=r'through $\mathsf{opt}_2$', **mylw)

p3, = ax.loglog(fgrid_12[0]-f1opt, fgrid_12[1]-f2opt, color=myc[2], ls=myls[2],
                label=r'through both optima', **mylw)


# Get Pareto front from vectors of objective values obtained
objs = np.vstack((fgrid_rand_1[0], fgrid_rand_1[1])).transpose()
pfFlag = pf.callParetoFront(objs)
ax.loglog(fgrid_rand_1[0][pfFlag]-f1opt, fgrid_rand_1[1][pfFlag]-f2opt, color=myc[0], ls='', marker='.', markersize=8, markeredgewidth=0,
                             alpha=0.4)
objs = np.vstack((fgrid_rand_2[0], fgrid_rand_2[1])).transpose()
pfFlag = pf.callParetoFront(objs)
ax.loglog(fgrid_rand_2[0][pfFlag]-f1opt, fgrid_rand_2[1][pfFlag]-f2opt, color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
                             alpha=0.4)
objs = np.vstack((fgrid_12[0], fgrid_12[1])).transpose()
pfFlag = pf.callParetoFront(objs)
ax.loglog(fgrid_12[0][pfFlag]-f1opt, fgrid_12[1][pfFlag]-f2opt, color=myc[2], ls='', marker='.', markersize=8, markeredgewidth=0,
                             alpha=0.4)

# plot nadir:
ax.loglog(nadir[0]-f1opt, nadir[1]-f2opt, color='k', ls='', marker='+', markersize=8, markeredgewidth=1.5,
                             alpha=0.9)

# beautify:
ax.set_xlabel(r'$f_1 - f_1^\mathsf{opt}$', fontsize=16)
ax.set_ylabel(r'$f_2 - f_2^\mathsf{opt}$', fontsize=16)
ax.legend(loc="best", framealpha=0.2)
ax.set_title("BBOB $f_{%d.%d}$ versus BBOB $f_{%d.%d}$ along three directions (%d-D)" % (f1_id, f1_instance, f2_id, f2_instance, dim))
[line.set_zorder(3) for line in ax.lines]
[line.set_zorder(3) for line in ax.lines]
fig.subplots_adjust(left=0.1) # more room for the y-axis label

# TODO: we might want to zoom in a bit:
#ax.set_xlim((0, 2*(nadir[0] - f1opt)))
#ax.set_ylim((0, 2*(nadir[1] - f2opt)))

# add rectangle as ROI
ax.add_patch(patches.Rectangle(
        (ideal[0]-f1opt + 1e-6, ideal[1]-f2opt + 1e-6), nadir[0]-ideal[0], nadir[1]-ideal[1],
        alpha=0.05,
        color='k'))

plt.show(block=True)







##############################################################
#                                                            #
# Plot the same, but not in log-scale.                       #
#                                                            #
##############################################################

fig = plt.figure(2)
ax = fig.add_subplot(111)


ax.plot(f_xopt1[0], f_xopt1[1], color='green', ls='', marker='*', markersize=8, markeredgewidth=0.5, markeredgecolor='black')
ax.plot(f_xopt2[0], f_xopt2[1], color='blue', ls='', marker='*', markersize=8, markeredgewidth=0.5, markeredgecolor='black')

p1, = ax.plot(fgrid_rand_1[0], fgrid_rand_1[1], color=myc[0], ls=myls[2],
                label=r'through $\mathsf{opt}_1$', **mylw)

p2, = ax.plot(fgrid_rand_2[0], fgrid_rand_2[1], color=myc[1], ls=myls[2],
                label=r'through $\mathsf{opt}_2$', **mylw)

p3, = ax.plot(fgrid_12[0], fgrid_12[1], color=myc[2], ls=myls[2],
                label=r'through both optima', **mylw)


# Get Pareto front from vectors of objective values obtained
#ax.plot(fgrid_rand_1[0][pfFlag], fgrid_rand_1[1][pfFlag], color=myc[0], ls='', marker='.', markersize=8, markeredgewidth=0,
#                             alpha=0.8)
#ax.plot(fgrid_rand_2[0][pfFlag], fgrid_rand_2[1][pfFlag], color=myc[1], ls='', marker='.', markersize=8, markeredgewidth=0,
#                             alpha=0.8)
#ax.plot(fgrid_12[0][pfFlag], fgrid_12[1][pfFlag], color=myc[2], ls='', marker='.', markersize=8, markeredgewidth=0,
#                             alpha=0.8)

# plot nadir:
#ax.plot(nadir[0], nadir[1], color='k', ls='', marker='+', markersize=8, markeredgewidth=1.5,
#                             alpha=0.9)
# plot ideal:
#ax.plot(ideal[0], ideal[1], color='k', ls='', marker='x', markersize=8, markeredgewidth=1.5,
#                             alpha=0.9)


# beautify:
ax.set_xlabel(r'$f_{%d.%d}$' % (f1_id, f1_instance), fontsize=16)
ax.set_ylabel(r'$f_{%d.%d}$' % (f2_id, f2_instance), fontsize=16)
ax.legend(loc="best", framealpha=0.2)
ax.set_title("BBOB $f_{%d.%d}$ versus BBOB $f_{%d.%d}$ along three directions (%d-D)" % (f1_id, f1_instance, f2_id, f2_instance, dim))
[line.set_zorder(3) for line in ax.lines]
[line.set_zorder(3) for line in ax.lines]
fig.subplots_adjust(left=0.1) # more room for the y-axis label

# zoom into Pareto front:
ax.set_xlim((ideal[0]-0.5*(nadir[0] - ideal[0]), nadir[0] + (nadir[0] - ideal[0])))
ax.set_ylim((ideal[1]-0.5*(nadir[1] - ideal[1]), nadir[1] + (nadir[1] - ideal[1])))


# add rectangle as ROI
ax.add_patch(patches.Rectangle(
        (ideal[0], ideal[1]), nadir[0]-ideal[0], nadir[1]-ideal[1],
        alpha=0.05,
        color='k'))








plt.show(block=True)
