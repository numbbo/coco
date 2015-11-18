'''
Generates "pprldistr2009_RLB.pickle.gz" that is used to plot, in the background, the results of the 2009 algorithms
'''
# datapath = "../../data-archive/data/gecco-bbob-1-24/2009/data"
datapaths = ["../../data-archive/data/gecco-bbob-1-24/2009/data", "../../data-archive/data/gecco-bbob-noisy/2009/data"]
savepath = "bbob_pproc/pprldistr2009_hardestRLB.pickle"
# savepath = "bbob_pproc/pprldistr2009_RLB.pickle"
import pickle
import bbob_pproc as bb
import numpy as np
data = {}
for datapath in datapaths:
    print "loading data from", datapath
    data2009 = bb.load(datapath)
    Algs = data2009.dictByAlg()
    target_runlengths_in_table = [0.5, 1.2, 3, 10, 50]
    targets = bb.pproc.RunlengthBasedTargetValues(target_runlengths_in_table,
                                                force_different_targets_factor = 10 ** -0.2)

    for alg in Algs:
        curAlg = Algs[alg].dictByFunc()
        algname = curAlg[curAlg.keys()[0]][0].algId
        if not data.has_key(algname):
            data[algname] = {}
        for func in curAlg:
            data[algname][func] = {}
            funcdata = curAlg[func].dictByDim()
            for dim in funcdata:
                data[algname][func][dim] = [[]]
                curtarget = targets((func, dim))[-1]
                data[algname][func][dim][0].append(curtarget) # record hardest target
                datum = funcdata[dim][0]
                y = datum.detEvals([curtarget])[0]
                data[algname][func][dim][0].append(y)
                x = y[np.isnan(y) == False]
                bb.pprldistr.plotECDF(x[np.isfinite(x)] / float(dim), len(y))
        print algname, "done"
with open(savepath, "w") as f:
    pickle.dump(data, f)
"""
G.detEvals([targets((G.funcId,G.dim))[-1]])
x=hstack(G.detEvals([targets((G.funcId,G.dim))[-1]]))
y=x[isnan(list(x))==False]
"""
