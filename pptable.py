#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for generating tables used by rungeneric1.py.

The generated tables give the ERT and in brackets the 10th to 90th
percentile range divided by two of 100 simulated runs divided by the
best ERT measured during BBOB-2009 (given in the respective first row)
for different target precisions for different functions. If no algorithm
in BBOB-2009 reached the target precision, the absolute values are
given.

The median number of conducted function evaluations is given in
*italics*, if no run reached 1e-7.
#succ is the number of trials that reached the target precision 1e-8
**Bold** entries are statistically significantly better (according to
the rank-sum test) compared to the best algorithm in BBOB-2009, with
p = 0.05 or p = 1e-k where k > 1 is the number following the
\downarrow symbol, with Bonferroni correction by the number of
functions.

"""
from __future__ import absolute_import

import os
import numpy as np
import matplotlib.pyplot as plt
from bbob_pproc import bestalg, toolsstats, pproc
from bbob_pproc.pptex import tableLaTeX, tableLaTeXStar, writeFEvals2, writeFEvalsMaxPrec
from bbob_pproc.toolsstats import significancetest

from pdb import set_trace

targets = (10., 1., 1e-1, 1e-3, 1e-5, 1e-7) # targets of the table
finaltarget = 1e-8 # value for determining the success ratio
targetsOfInterest = (10., 1., 1e-1, 1e-3, 1e-5, 1e-7) # targets of the table
targetsOfInterest = pproc.TargetValues((10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7))
targetf = 1e-8 # value for determining the success ratio
samplesize = 1000 # TODO: change samplesize
# def tablespec(targets):
# 
#     i = 0
#     tspec = {'col%d' % i: {'what': 'fname', 'header': r'$\Delta f$', 'format': None}}
#     for t in targets:
#         i =  i + 1
#         tspec.update({'col%d' % i: {'what': 'ERT ratio for df=%e' % t,
#                                     'header': r'\multicolumn{2}{@{}c@{}}{1e%+d}' % (int(np.log10(t)),
#                                     'format': writeFEval}})
#     i = i + 1
#     tspec.update({'col%d' %  i: {'what': 'nb of success', 'header': r'\#succ',
#                                  'format': '%d'}})
    
    
old_legend = r""" 
 \newcommand{\tablecaption}[1]{Shown are, for functions #1 and for a
 given target difference to the optimal function value \Df: the number
 of successful trials (\textbf{$\#$}); the expected running time to
 surpass $\fopt+\Df$ (\ERT, see Figure~\ref{fig:ERTgraphs}); the
 \textbf{10\%}-tile and \textbf{90\%}-tile of the bootstrap
 distribution of \ERT; the average number of function evaluations in
 successful trials or, if none was successful, as last entry the median
 number of function evaluations to reach the best function value
 ($\text{RT}_\text{succ}$).  If $\fopt+\Df$ was never reached, figures in
 \textit{italics} denote the best achieved \Df-value of the median
 trial and the 10\% and 90\%-tile trial.  Furthermore, N denotes the
 number of trials, and mFE denotes the maximum of number of function
 evaluations executed in one trial. See Figure~\ref{fig:ERTgraphs} for
 the names of functions. }
"""

table_caption=r"""Expected running time (ERT in number of function evaluations)
divided by the best ERT measured during BBOB-2009 (given in the
first row of each cell) for different $\Df$ values for functions
$f_1$--$f_{24}$.
The median number of conducted function evaluations is additionally given in 
\textit{italics}, if $\ERT(10^{-7}) =\infty$. The interquartile range divided by two is shown in braces. 
%
\#succ is the number of trials that reached the final target $\fopt + 10^{-8}$.
"""

table_caption_rlbased=r"""Expected running time (ERT in number of function evaluations) to reach the run 
length based target values (left side of ":" in the first row of each cell) obtained using
the best ERT measured during BBOB-2009 (right side of ":"
in first row of each cell) divided by the corresponding ERT for different run lengths for functions
$f_1$--$f_{24}$.
The median number of conducted function evaluations is additionally given in 
\textit{italics}, if $\ERT(10^{-7}) =\infty$. The interquartile range divided by two is shown in braces. 
%
\#succ is the number of trials that reached the final target $\fopt + 10^{-8}$.
"""


def _treat(ds):
    # Rec array: http://docs.scipy.org/doc/numpy/user/basics.rec.html
    bestentry = bestalg.bestalgentries2009[(ds.dim, ds.funcId)]
    bestert = bestentry.detERT(targets)
    bestevals, bestalgs = bestentry.detEvals(targets)
    bestfinaldata = bestentry.detEvals([finaltarget])[0][0]
    ert = ds.detERT(targets)
    evals = ds.detEvals(targets)
    finaldata = ds.detEvals([finaltarget])[0]

    dtype = []
    bestdtype = []
    for i, t in enumerate(targets):
        dtype.append((('ERT ratio (iq 10-90), df=%e' % t, 'df=%e' % t), '2f')) 
        bestdtype.append((('best ERT df=%e' % t, 'df=%e' % t), 'f'))
    dtype.append((('nb success final target=%e' % t, 'finaltarget=%e' % t), 'i8'))
    dtype.append(('nbruns', 'i8'))
    bestdtype.append((('nb success finaltarget=%e' % t, 'finaltarget=%e' % t), 'i8'))
    bestdtype.append(('nbruns', 'i8'))
    besttable = np.zeros(1, dtype=bestdtype)
    wholetable = np.zeros(1, dtype=dtype)
    table = wholetable[0]
    
    bestdata = list()
    bestdata.extend(bestert)
    bestdata.append(np.sum(np.isnan(bestfinaldata) == False))
    bestdata.append(len(bestfinaldata))
    besttable[0] = tuple(bestdata)

    data = list()
    for i, e in enumerate(evals): # loop over targets
        unsucc = np.isnan(e)
        bt = toolsstats.drawSP(e[unsucc == False], ds.maxevals[unsucc],
                               (10, 90), samplesize)[0]
        data.append((ert[i] / bestert[i], (bt[-1] - bt[0]) / 2. / bestert[i]))
    data.append(np.sum(np.isnan(finaldata) == False))
    data.append(ds.nbRuns())
    table = tuple(data) # fill with tuple not list nor array!

    # TODO: do the significance test thing here.
    return besttable, wholetable

def _table(data):
    res = []
    
    return res

def main2(dsList, dimsOfInterest, outputdir='.', info='', verbose=True):
    """Generate a table of ratio ERT/ERTbest vs target precision.
    
    1 table per dimension will be generated.

    Rank-sum tests table on "Final Data Points" for only one algorithm.
    that is, for example, using 1/#fevals(ftarget) if ftarget was
    reached and -f_final otherwise as input for the rank-sum test, where
    obviously the larger the better.

    """
    # TODO: remove dimsOfInterest, was added just for compatibility's sake
    if info:
        info = '_' + info
        # insert a separator between the default file name and the additional
        # information string.
    
    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()
    for d, dsdim in dsList.dictByDim().iteritems():
        dictfun = dsdim.dictByFunc()
        res = []
        for f, dsfun in sorted(dsdim.dictByFunc().iteritems()):
            assert len(dsfun) == 1, ('Expect one-element DataSetList for a '
                                     'given dimension and function')
            ds = dsfun[0]
            data = _treat(ds)
            res = _table(data)
        res = []
        outputfile = os.path.join(outputdir, 'pptable_%02dD%s.tex' % (d, info))
        f = open(outputfile, 'w')
        f.write(res)
        f.close()
        if verbose:
            print "Table written in %s" % outputfile

def main(dsList, dimsOfInterest, outputdir, info='', verbose=True):
    """Generate a table of ratio ERT/ERTbest vs target precision.
    
    1 table per dimension will be generated.

    Rank-sum tests table on "Final Data Points" for only one algorithm.
    that is, for example, using 1/#fevals(ftarget) if ftarget was
    reached and -f_final otherwise as input for the rank-sum test, where
    obviously the larger the better.

    """
    #TODO: check that it works for any reference algorithm?
    #in the following the reference algorithm is the one given in
    #bestalg.bestalgentries which is the virtual best of BBOB
    dictDim = dsList.dictByDim()

    if info:
        info = '_' + info
        # insert a separator between the default file name and the additional
        # information string.

    dims = set(dictDim.keys())
    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()
    if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
        header = [r'\#FEs/D']
        for i in targetsOfInterest.labels():
            header.append(r'\multicolumn{2}{@{}c@{}}{%s}'
                      % i) #writeFEvals2(round(10**float(i)*500,2),2)) 
    else:
        header = [r'$\Delta f$']
        for i in targetsOfInterest.target_values:
            header.append(r'\multicolumn{2}{@{}c@{}}{1e%+d}'
                      % (int(np.log10(i))))
    header.append(r'\multicolumn{2}{|@{}r@{}}{\#succ}')

    for d in dimsOfInterest:
        table = [header]
        extraeol = [r'\hline']
        try:
            dictFunc = dictDim[d].dictByFunc()
        except KeyError:
            continue
        funcs = set(dictFunc.keys())
        nbtests = float(len(funcs)) # #funcs tests times one algorithm

        for f in sorted(funcs):
            bestalgentry = bestalg.bestalgentries2009[(d, f)]
            curline = [r'${\bf f_{%d}}$' % f]
            bestalgdata = bestalgentry.detERT(targetsOfInterest((f,d)))
            bestalgevals, bestalgalgs = bestalgentry.detEvals(targetsOfInterest((f,d)))
            if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
                #write ftarget:fevals
                for i in xrange(len(bestalgdata[:-1])):
                    temp="%.1e" %targetsOfInterest((f,d))[i]
                    if temp[-2]=="0":
                        temp=temp[:-2]+temp[-1]
                    curline.append(r'\multicolumn{2}{@{}c@{}}{\textit{%s}:%s \quad}'
                                   % (temp,writeFEvalsMaxPrec(bestalgdata[i], 2)))
                temp="%.1e" %targetsOfInterest((f,d))[-1]
                if temp[-2]=="0":
                    temp=temp[:-2]+temp[-1]
                curline.append(r'\multicolumn{2}{@{}c@{}|}{\textit{%s}:%s }'
                               % (temp,writeFEvalsMaxPrec(bestalgdata[-1], 2)))            
            else:            
                # write #fevals of the reference alg
                for i in bestalgdata[:-1]:
                    curline.append(r'\multicolumn{2}{@{}c@{}}{%s \quad}'
                                   % writeFEvalsMaxPrec(i, 2))
                curline.append(r'\multicolumn{2}{@{}c@{}|}{%s}'
                               % writeFEvalsMaxPrec(bestalgdata[-1], 2))
    


            # write the success ratio for the reference alg
            tmp = bestalgentry.detEvals([targetf])[0][0]
            tmp2 = np.sum(np.isnan(tmp) == False) # count the nb of success
            curline.append('%d' % (tmp2))
            if tmp2 > 0:
                curline.append('/%d' % len(tmp))

            table.append(curline[:])
            extraeol.append('')

            # generate all data for ranksum test
            assert len(dictFunc[f]) == 1
            entry = dictFunc[f][0] # take the first element
            ertdata = entry.detERT(targetsOfInterest((f,d)))

            testresbestvs1 = significancetest(bestalgentry, entry, targetsOfInterest((f,d)))

            #for nb, entry in enumerate(entries):
            #curline = [r'\algshort\hspace*{\fill}']
            curline = ['']
            #data = entry.detERT(targetsOfInterest)
            evals = entry.detEvals(targetsOfInterest((f,d)))
            dispersion = []
            data = []
            for i in evals:
                succ = (np.isnan(i) == False)
                tmp = i.copy()
                tmp[succ==False] = entry.maxevals[np.isnan(i)]
                #set_trace()
                data.append(toolsstats.sp(tmp, issuccessful=succ)[0])
                #if not any(succ):
                    #set_trace()
                if any(succ):
                    tmp2 = toolsstats.drawSP(tmp[succ], tmp[succ==False],
                                            (10, 50, 90), samplesize)[0]
                    dispersion.append((tmp2[-1]-tmp2[0])/2.)
                else:
                    dispersion.append(None)

            for i, ert in enumerate(data):
                alignment = 'c'
                if i == len(data) - 1: # last element
                    alignment = 'c|'

                nbstars = 0
                z, p = testresbestvs1[i]
                if ert - bestalgdata[i] < 0. and not np.isinf(bestalgdata[i]):
                    evals = entry.detEvals([targetsOfInterest((f,d))[i]])[0] 
                    evals[np.isnan(evals)] = entry.maxevals[np.isnan(evals)]
                    bestevals = bestalgentry.detEvals([targetsOfInterest((f,d))[i]])
                    bestevals, bestalgalg = (bestevals[0][0], bestevals[1][0])
                    bestevals[np.isnan(bestevals)] = bestalgentry.maxevals[bestalgalg][np.isnan(bestevals)]
                    evals = np.array(sorted(evals))[0:min(len(evals), len(bestevals))]
                    bestevals = np.array(sorted(bestevals))[0:min(len(evals), len(bestevals))]

                #The conditions are now that ERT < ERT_best and
                # all(sorted(FEvals_best) > sorted(FEvals_current)).
                if ((nbtests * p) < 0.05 and ert - bestalgdata[i] < 0.
                    and z < 0.
                    and (np.isinf(bestalgdata[i])
                         or all(evals < bestevals))):
                    nbstars = -np.ceil(np.log10(nbtests * p))

                if np.isinf(bestalgdata[i]): # if the best did not solve the problem
                    isBold = False
                    if nbstars > 0:
                       isBold = True

                    tmp = writeFEvalsMaxPrec(float(ert), 2)
                    if not np.isinf(ert):
                        tmp = r'\textit{%s}' % (tmp)
                        if isBold:
                            tmp = r'\textbf{%s}' % tmp

                    tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                  % (alignment, tmp))
                else:
                    # Formatting
                    tmp = float(ert)/bestalgdata[i]
                    assert not np.isnan(tmp)
                    tableentry = writeFEvalsMaxPrec(tmp, 2)

                    isBold = False
                    if nbstars > 0:
                       isBold = True

                    if np.isinf(tmp) and i == len(data)-1:
                        tableentry = (tableentry
                                      + r'\textit{%s}' % writeFEvals2(np.median(entry.maxevals), 2))
                        if isBold:
                            tableentry = r'\textbf{%s}' % tableentry
                        elif 11 < 3: # and significance0vs1 < 0:
                            tableentry = r'\textit{%s}' % tableentry
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                      % (alignment, tableentry))
                    elif tableentry.find('e') > -1 or (np.isinf(tmp) and i != len(data) - 1):
                        if isBold:
                            tableentry = r'\textbf{%s}' % tableentry
                        elif 11 < 3: # and significance0vs1 < 0:
                            tableentry = r'\textit{%s}' % tableentry
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                      % (alignment, tableentry))
                    else:
                        tmp = tableentry.split('.', 1)
                        if isBold:
                            tmp = list(r'\textbf{%s}' % i for i in tmp)
                        elif 11 < 3: # and significance0vs1 < 0:
                            tmp = list(r'\textit{%s}' % i for i in tmp)
                        tableentry = ' & .'.join(tmp)
                        if len(tmp) == 1:
                            tableentry += '&'

                superscript = ''

                if nbstars > 0:
                    #tmp = '\hspace{-.5ex}'.join(nbstars * [r'\star'])
                    if z > 0:
                        superscript = r'\uparrow' #* nbstars
                    else:
                        superscript = r'\downarrow' #* nbstars
                        # print z, linebest[i], line1
                    if nbstars > 1:
                        superscript += str(int(nbstars))

                #if superscript or significance0vs1:
                    #s = ''
                    #if significance0vs1 > 0:
                       #s = '\star'
                    #if significance0vs1 > 1:
                       #s += str(significance0vs1)
                    #s = r'$^{' + s + superscript + r'}$'

                    #if tableentry.endswith('}'):
                        #tableentry = tableentry[:-1] + s + r'}'
                    #else:
                        #tableentry += s

                if dispersion[i] and not np.isinf(bestalgdata[i]):
                    tmp = writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                    tableentry += (r'${\scriptscriptstyle(%s)}$' % tmp)

                if superscript:
                    s = r'$^{' + superscript + r'}$'

                    if tableentry.endswith('}'):
                        tableentry = tableentry[:-1] + s + r'}'
                    else:
                        tableentry += s

                curline.append(tableentry)

                #curline.append(tableentry)
                #if dispersion[i] is None or np.isinf(bestalgdata[i]):
                    #curline.append('')
                #else:
                    #tmp = writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 2)
                    #curline.append('(%s)' % tmp)

            tmp = entry.evals[entry.evals[:, 0] <= targetf, 1:]
            try:
                tmp = tmp[0]
                curline.append('%d' % np.sum(np.isnan(tmp) == False))
            except IndexError:
                curline.append('%d' % 0)
            curline.append('/%d' % entry.nbRuns())

            table.append(curline[:])
            extraeol.append(r'\hline')
        extraeol[-1] = ''

        outputfile = os.path.join(outputdir, 'pptable_%02dD%s.tex' % (d, info))
        if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
            spec = r'@{}c@{}|' + '*{%d}{@{ }r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
        else:
            spec = r'@{}c@{}|' + '*{%d}{@{}r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
        #res = r'\providecommand{\algshort}{%s}' % alg1 + '\n'
        #res += tableLaTeXStar(table, width=r'0.45\textwidth', spec=spec,
                              #extraeol=extraeol)
        res = tableLaTeX(table, spec=spec, extraeol=extraeol)
        f = open(outputfile, 'w')
        f.write(res)
        f.close()
        if verbose:
            print "Table written in %s" % outputfile
