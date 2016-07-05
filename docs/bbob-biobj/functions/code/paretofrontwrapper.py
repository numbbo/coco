
import numpy
import ctypes # from ctypes import *

_libmyCfuncs = numpy.ctypeslib.load_library('libparetofront', '.')
# _libmyCfuncs = ctypes.cdll.LoadLibrary('libparetofront.dll')

_libmyCfuncs.paretofront.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.bool),
                                     numpy.ctypeslib.ndpointer(dtype=numpy.double),
                                     ctypes.c_uint,
                                     ctypes.c_uint]
_libmyCfuncs.paretofront.restype  = ctypes.c_void_p


def callParetoFront(obj, nobj=None, nsample=None):
    obj = numpy.asarray(obj, dtype=numpy.double, order='F') # paretofront.c is compatible with MATLAB --> order='F'
    if nobj is None:
        nobj = obj.shape[1]
    if nsample is None:
        nsample = obj.shape[0]
    frontFlag = numpy.empty(nsample, dtype=numpy.bool) # prepare the output
    frontFlag[:] = False # BE CAREFUL: it goes wrong if not set to False!
    _libmyCfuncs.paretofront(frontFlag, obj, long(nsample), long(nobj))
    return frontFlag

# Test case:
# A = np.array([[1,4], [2,2], [4,1], [3,3], [2,3], [3,2], [1.5,3], [5,4], [3,1.5]])
# n = 100; P = np.random.randn(n,2); b = paretofrontwrapper.callParetoFront(P); plt.plot(P[:,0], P[:,1],'o'); plt.plot(P[b,0], P[b,1], 'o', color='red'); plt.show()
