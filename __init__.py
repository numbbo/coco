#!/usr/bin/env python
"""Given input files from BBOB experiments, postprocesses and returns output.

Keyword arguments:
    
"""

#credits to G. Van Rossum: http://www.artima.com/weblogs/viewpost.jsp?thread=4829

#Should load either from index files or from the post processed data files.
#Problem is: how do we get index entry information from the post processed data files?
#Add a commented header, and so shouldn't we store many post processed data into one file?

#TODO:

from __future__ import absolute_import

import sys
import os
import getopt
from bbob_pproc import readindexfiles
from bbob_pproc import findindexfiles
from bbob_pproc import pproc
from bbob_pproc import ppfig
from bbob_pproc import pptex
from bbob_pproc import ppdatapr
from pdb import set_trace
import scipy
import numpy as n
import matplotlib.pyplot as plt
import pickle

plt.rc("axes", labelsize=16, titlesize=16)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("font", size=16)
plt.rc("legend", fontsize=16)

__all__  = ['readindexfiles','findindexfiles','ppfig','pptex','ppdatapr',
            'main']
sp1index = 1
medianindex = 5
colors = {2:'b', 3:'g', 5:'r', 10:'c', 20:'m', 40:'y'} #TODO colormaps!
dimOfInterest = [5,20]    # dimension which are displayed in the tables

#Either we read it in a file (flexibility) or we hard code it here.
funInfos = {}
# SEPARABLE
funInfos[1]  =  '1 sphere';
funInfos[2]  =  '2 ellipsoid separable';
funInfos[3]  =  '3 Rastrigin separable';
funInfos[4]  =  '4 skew Rastrigin-Bueche separable';

# LOW OR MODERATE CONDITION
funInfos[5]  =  '5 linear slope';
funInfos[6]  =  '6 step-ellipsoid';
funInfos[7]  =  '7 Rosenbrock non-rotated';
funInfos[8]  =  '8 Rosenbrock rotated';

# HIGH CONDITION
funInfos[9]  =  '9 ellipsoid';
funInfos[10] = '10 discus';
funInfos[11] = '11 bent cigar';
funInfos[12] = '12 sharp ridge';
funInfos[13] = '13 sum of different powers';

# MULTI-MODAL
funInfos[14] = '14 Rastrigin';
funInfos[15] = '15 skew Rastrigin-Bueche';
funInfos[16] = '16 Weierstrass';
funInfos[17] = '17 Schaffer F7, condition 10';
funInfos[18] = '18 Schaffer F7, condition 1000';
funInfos[19] = '19 F8F2';

# MULTI-MODAL WITH WEAK GLOBAL STRUCTURE
funInfos[20] = '20 Schwefel x*sin(x)';
funInfos[21] = '21 Gallagher, global rotation';
funInfos[22] = '22 Gallagher, local rotations';
funInfos[23] = '23 Katsuuras';
funInfos[24] = '24 Lunacek bi-Rastrigin';

# MODERATE NOISE
funInfos[101] = '101 sphere moderate noise1';
funInfos[102] = '102 sphere moderate noise2';
funInfos[103] = '103 sphere moderate noise3';
funInfos[104] = '104 Rosenbrock non-rotated moderate noise1';
funInfos[105] = '105 Rosenbrock non-rotated moderate noise2';
funInfos[106] = '106 Rosenbrock non-rotated moderate noise3';

# SEVERE NOISE
funInfos[107] = '107 sphere noise1';
funInfos[108] = '108 sphere noise2';
funInfos[109] = '109 sphere noise3';
funInfos[110] = '110 ellipsoid noise1';
funInfos[111] = '111 ellipsoid noise2';
funInfos[112] = '112 ellipsoid noise3';
funInfos[113] = '113 step-ellipsoid noise1';
funInfos[114] = '114 step-ellipsoid noise2';
funInfos[115] = '115 step-ellipsoid noise3';
funInfos[116] = '116 Rosenbrock non-rotated noise1';
funInfos[117] = '117 Rosenbrock non-rotated noise2';
funInfos[118] = '118 Rosenbrock non-rotated noise3';
funInfos[119] = '119 sum of different powers noise1';
funInfos[120] = '120 sum of different powers noise2';
funInfos[121] = '121 sum of different powers noise3';

# SEVERE NOISE HIGHLY MULTI-MODAL
funInfos[122] = '122 Schaffer F7 noise1';
funInfos[123] = '123 Schaffer F7 noise2';
funInfos[124] = '124 Schaffer F7 noise3';
funInfos[125] = '125 F8F2 noise1';
funInfos[126] = '126 F8F2 noise2';
funInfos[127] = '127 F8F2 noise3';
funInfos[128] = '128 Gallagher noise1';
funInfos[129] = '129 Gallagher noise2';
funInfos[130] = '130 Gallagher noise3';

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#Should take as argument paths to info files (since it is what is read by read index files,
# or post-processed data files to avoid processing raw data every time.
def usage():
    print "python bbob_pproc.py -v" #TODO

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:

        try:
            opts, args = getopt.getopt(argv[1:], "hvfo:",
                                       ["help","files","output-dir",
                                        "tab-only","fig-only","dat-only"])
        except getopt.error, msg:
             raise Usage(msg)
        # more code, unchanged
        outputdir = 'ppdata'
        verbose = True #TODO put verbose function everywhere needed.
        isfigure = True
        istab = True
        isdataprof = True
        isPostProcessed = False

        for o, a in opts:
            if o == "-v":
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-f", "--files"):
                #Post processed data files
                isPostProcessed = True
            elif o in ("-o", "--output-dir"):
                outputdir = a
            #The next 2 are for testing purpose
            elif o == "--tab-only":
                isfigure = False
                isdataprof = False
            elif o == "--fig-only":
                istab = False
                isdataprof = False
            elif o == "--dat-only":
                istab = False
                isfigure = False
            else:
                assert False, "unhandled option"

        if isPostProcessed:
            indexEntries = []
            for i in args:
                if i.endswith('.pickle'):
                    f = open(i,'r')
                    indexEntries.append(pickle.load(f))
                    f.close()
                    if verbose:
                        print 'Unpickled %s.' %(i)
        else:
            indexFiles = []
            for i in args:
                if i.endswith('.info'):
                    (filepath,filename) = os.path.split(i)
                    indexFiles.append(findindexfiles.IndexFile(filepath,filename))
                else:
                    indexFiles.extend(findindexfiles.main(i,verbose))
                #Todo watch that the same info file is not listed twice this way.
            indexEntries = readindexfiles.main(indexFiles,verbose)

        sortByFunc = {}
        #set_trace()
        for elem in indexEntries:
            if not sortByFunc.has_key(elem.funcId): #This takes a while!
                sortByFunc[elem.funcId] = {}
            sortByFunc[elem.funcId][elem.dim] = elem
            if not isPostProcessed:
                pproc.main(elem,verbose)
                filename =os.path.join(outputdir,
                                       'ppdata_f%d_%d' %(elem.funcId,elem.dim))
                f = open(filename + '.pickle','w')
                pickle.dump(elem,f)
                f.close()
                if verbose:
                    print 'Pickle in %s.' %(filename+'.pickle')

        #set_trace()
        # create directory

        #filename2 = 'ppdata/ppdata'

        #if not(isfigure or istab):
            #return None
        try:
            os.listdir(os.getcwd()).index(outputdir)
        except ValueError:
            os.mkdir(outputdir)

        ps = []
        nbRuns = []

        for func in sortByFunc.keys():
            filename = os.path.join(outputdir,'ppdata_f%d' % (func))
            tabData = n.zeros(0)   # initialize matrix containing the table entries
            entryList = list()     # list of entries which are displayed in the table
            if isfigure:
                fig = plt.figure()

            for dim in sorted(sortByFunc[func].keys()):
                entry = sortByFunc[func][dim]
                #pproc.main(entry,verbose)
                #set_trace()
                ps.append(entry.arrayTab[-1,4])
                nbRuns.append(entry.nbRuns)

                if dimOfInterest.count(dim) != 0 and tabData.shape[0] == 0:
                    # Array tabData has no previous values.
                    tabData = entry.arrayTab
                    entryList.append(entry)
                elif dimOfInterest.count(dim) != 0 and tabData.shape[0] != 0:
                    # Array tabData already contains values for the same function
                    tabData = n.append(tabData,entry.arrayTab[:,1:],1)
                    entryList.append(entry)

                if isfigure:
                    #set_trace()
                    h=ppfig.createFigure(entry.arrayFullTab[:,[sp1index,0]],
                                           fig)
                    for i in h:
                        plt.setp(i,'color',colors[dim])

                    h=ppfig.createFigure(entry.arrayFullTab[:,[medianindex,0]],
                                         fig)
                    for i in h:
                        plt.setp(h,'color',colors[dim],'linestyle','--')

            if istab:
                [header,format] = pproc.computevalues(None,None,header=True)
                pptex.writeTable2(tabData, filename, entryList,
                                  fontSize = 'tiny', header = header,
                                  format = format, verbose = verbose)

            if isfigure:
                #set_trace()
                ppfig.customizeFigure(fig, filename,
                                      title = funInfos[entry.funcId],
                                      fileFormat = ('eps','png'),
                                      labels = ['',''],
                                      scale = ['log','log'],
                                      locLegend = 'best',verbose = verbose)
                #plt.show(fig)
                plt.close(fig)

        if isdataprof:
            sortForDataProf = ppdatapr.sortIndexEntries(indexEntries)
            valuesOfInterest = [1.0,1.0e-2,1.0e-4,1.0e-6,1.0e-8]
            colorsDataProf = ['b', 'g', 'r', 'c', 'm']
            #synchronize this with the valuesForDataProf in pproc

            for i in range(len(sortForDataProf)):
                figureName = os.path.join(outputdir,'ppdataprofile%d' %(i))
                i = sortForDataProf[i]
                fig = plt.figure()
                for j in range(len(valuesOfInterest)):
                #for j in [0]:
                    maxEvalsFactor = 1e4
                    tmp = ppdatapr.main(i,valuesOfInterest[j],maxEvalsFactor,
                                        verbose)
                    if not tmp is None:
                        plt.setp(tmp,'color',colorsDataProf[j])
                ppdatapr.beautify(fig, figureName, maxEvalsFactor,
                                  verbose = verbose)
                plt.close(fig)

        if verbose:
            #set_trace()
            print 'total ps = %g\n' % (float(scipy.sum(ps))/scipy.sum(nbRuns))

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
   sys.exit(main())
