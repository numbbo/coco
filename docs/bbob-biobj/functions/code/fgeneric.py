#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test function interface.

This module implements class LoggingFunction which is the main class for
logging a whole experiment.

Example: 

    >>> import fgeneric, bbobbenchmarks as bb
    >>> f = fgeneric.LoggingFunction('fgeneric_doctest_no_algorihm', 'no-name-algorithm')  # like a LoggingFunction class 
    >>> f.setfun(bb.F1()).ftarget  # assumes that F1().fopt is accessible, prints target value
    -92.649999990000012
    >>> f([1,2,3])  # call f1 while logging the results, same as f.evalfun(x) 
    -61.233399680000005
    >>> f.finalizerun()  # reset counter for f-evaluations etc

For other examples of how to use fgeneric, see exampleexperiment.py and exampletiming.py

Test:
    
    python fgeneric.py 

runs doctest, using -v option gives verbosity.  

"""

# Changes: 
# 12/03/08: renamed cocoexp.Logger to fgeneric.LoggingFunction and setfun returns self
# 12/02/22: def setfun, the second argument has become optional and attribute  
#           fopt of the first argument is used, if the second is not given 

import sys
import os
import errno
import warnings
from pdb import set_trace
import copy
import numpy as np

deltaftarget = 1e-8
nb_evaluations_always_written = '1'  # '100 + 10 * dim'  # 100 + dim;10*dim add about 7;17MB to final data
nb_triggers_per_delta_f_decade = 5   # using 10 should be perfectly fine 
fileprefix = 'bbobexp'

class LoggingFunction(object):
    """Class for a function that records data from experiments with a given 
       algorithm and parameter settings.
    
    This class provides recording facilities:
    
    * index files (:file:`info` extension) record information on an
      experiment
    * data files (:file:`dat`, :file:`tdat` extensions) which record
      number of function evaluations and function values for consecutive
      runs.

    Once an instance of this class is provided with a test function and
    the function optimal value, the instance or its method :meth:`evalfun` 
    can be called to evaluate the function.

    At the end of the run, :meth:`finalizerun` must be called.

    Either the provided test function is an object with an evaluation
    method :meth:`_evalfull` which returns noisy and noiseless values
    (tuple of length 2). Otherwise, the test function should be
    callable and return a scalar if called with tuple (0., 0.).

    """
    nbptsevals = 20. # number of trigger per decade of function evaluations
    nbptsf = float(nb_triggers_per_delta_f_decade) # number of trigger per decade of function values, float to prevent integer division

    def __call__(self, x, *args, **kwargs): # makes the instances callable
        """Returns objective function value"""
        return self.evalfun(x, *args, **kwargs)

    class __Eval(object):
        """Class for the object lasteval."""
        def __init__(self):
            # TODO: check what is actually needed here.
            self.num = 0.
            self.f = np.inf # minimization...
            self.bestf = np.inf # minimization...
            self.fnoisy = np.inf # minimization...
            self.bestfnoisy = np.inf # minimization...
            self.is_written = False
        def sprintData(self, fopt):
            """Format data for printing."""
    
            res = ('%d %+10.9e %+10.9e %+10.9e %+10.9e'
                   % (self.num, self.f - fopt,
                      self.bestf - fopt, self.fnoisy,
                      self.bestfnoisy))
    
            if len(self.x) < 22:
                tmp = []
                for i in self.x:
                    tmp.append(' %+5.4e' % i)
                res += ''.join(tmp)
            res += '\n'
            return res

        def update(self, fnoisy, f, x):
            """Update the content of lasteval."""
            try:
                self.num += len(f)
                self.f = f[-1]
                self.fnoisy = fnoisy[-1]
                self.x = x[-1]
            except TypeError:
                self.num += 1
                self.f = f
                self.fnoisy =  fnoisy
                self.x = x
            bestf = np.min(f)
            if bestf < self.bestf:
                self.bestf = bestf
            bestfnoisy = np.min(fnoisy)
            if bestfnoisy < self.bestfnoisy:
                self.bestfnoisy = bestfnoisy
            self.is_written = False

    def __init__(self, datapath, algid='not-specified', comments='',
                 inputformat='row'):
        """Initialize LoggingFunction for an experiment. Before the 
        LoggingFunction can be used as an objective
        function, method :meth:`setfun` must be called.
        
        :param string datapath: Output folder name
        :param string algid: name of algorithm
        :param string comments: complementary information on experiment
          (parameter settings and such)
        :param string inputformat: 'row' (default) or 'col', determines
          the shape of the input data.
        
        """
        self.initialize(datapath, algid, comments, inputformat)

    def initialize(self, datapath, algid='not-specified', comments='',
                   inputformat='row'):
        """Initialize LoggingFunction with a data path and further 
        infos. Before the LoggingFunction can be used as an objective
        function, method :meth:`setfun` must be called. 
        
        """
        self.datapath = datapath
        self.algid = algid
        self.comments = comments
        self.inputformat = inputformat
        self.fileprefix = fileprefix
        self._is_setfun = False
        self._is_setdim = False
        self._is_finalized = True
        self._is_samefun = False
        self._is_samedim = False

    def __del__(self):
        """Destructor.

        Will attempt to finalize a run when the current instance is
        about to be deleted.

        """
        try:
            if not self._is_finalized:
                self.finalizerun()
        except AttributeError:
            pass

    def _is_ready(self):
        res = (self._is_setdim and self._is_setfun and not self._is_finalized)
        return res

    def _readytostart(self):
        # index entry, data files
        filename = '%s_f%s_DIM%d' % (self.datafileprefix,
                                     str(self.funId), self._dim)

        res = []
        if (not (self._is_samefun and self._is_samedim)
            or not os.path.exists(self.indexfile)):
            i = 0
            while os.path.exists(filename + '.dat'):
                i += 1
                filename = '%s-%02d_f%s_DIM%d' % (self.datafileprefix, i,
                                                  str(self.funId), self._dim)
            self.datafile = filename + '.tdat'
            self.hdatafile = filename + '.dat'
            self.rdatafile = filename + '.rdat'

            if os.path.exists(self.indexfile):
                res.append('\n')  # was: res += '\n'

            if isinstance(self.funId, str):
                tmp = "'%s'" % self.funId
            else:
                tmp = str(self.funId)
            res.append('funcId = %s' % tmp)
            for i in self._fun_kwargs.iteritems():
                if isinstance(i[1], str):
                    tmp = "'%s'" % i[1]
                else:
                    tmp = str(i[1])
                res.append(', %s = %s' % (str(i[0]), i[1]))
            res.append(', DIM = %d, Precision = %.3e, algId = \'%s\'\n'
                       % (self._dim, self.precision, self.algid))
            res.append('%% %s\n%s' % (self.comments,
                               os.path.relpath(self.hdatafile, self.datapath)))

        if isinstance(self.iinstance, str):
            tmp = "'%s'" % self.iinstance
        else:
            tmp = str(self.iinstance)
        res.append(', %s' % tmp)
        f = open(self.indexfile, 'a')
        f.writelines(res) # does not add line separator
        f.close()
        self._is_finalized = False # Just starting.

        for datafile in (self.datafile, self.hdatafile, self.rdatafile):
            filepath, filename = os.path.split(datafile)
            try:
                os.makedirs(filepath)
            except OSError as (err, strerror):
                if err == errno.EEXIST:
                    pass
                else:
                    print errno, strerror
            f = open(datafile, 'a')
            f.write('%% function evaluation | noise-free fitness - Fopt'
                    ' (%13.12e) | best noise-free fitness - Fopt | measured '
                    'fitness | best measured fitness | x1 | x2...\n'
                    % self.fopt)
            f.close()

    def evalfun(self, inputx, *args, **kwargs):
        """Evaluate the function, return objective function value. 
        
        Positional and keyword arguments args and kwargs are directly
        passed to the test function evaluation method.
        
        """
        # This block is the opposite in Matlab!
        if self._is_rowformat:
            x = np.asarray(inputx)
        else:
            x = np.transpose(inputx)
            
        curshape = np.shape(x)
        dim = curshape[-1]
        if len(curshape) < 2:
            popsi = 1
        else:
            popsi = curshape[0]

        if not self._is_setdim or self._dim != dim:
            self._setdim(dim)

        if not self._is_ready():
            self._readytostart()

        out = self._fun_evalfull(x, *args, **kwargs)
        try:
            fvalue, ftrue = out
        except TypeError:
            fvalue = out
            ftrue = out
            self._fun_eval = self._fun_evalfull
            self._fun_evalfull = (lambda x: tuple([self._fun_eval(x)] * 2))

        if (self.lasteval.num + popsi >= self.evalsTrigger or
            np.min(ftrue) - self.fopt < self.fTrigger): # need to write something
            buffr = []
            hbuffr = []
            for j in range(0, popsi):
                try: 
                    fvaluej = fvalue[j]
                    ftruej = ftrue[j]
                    xj = x[j]
                except (IndexError, ValueError, TypeError): # cannot slice a 0-d array
                    fvaluej = fvalue
                    ftruej = ftrue
                    xj = x
                self.lasteval.update(fvaluej, ftruej, xj)

                if self.lasteval.num >= self.evalsTrigger:
                    buffr.append(self.lasteval.sprintData(self.fopt))
                    while self.lasteval.num >= np.floor(10**(self.idxEvalsTrigger/self.nbptsevals)):
                        self.idxEvalsTrigger += 1
                    while self.lasteval.num >= dim * 10**self.idxDIMEvalsTrigger:
                        self.idxDIMEvalsTrigger += 1
                    self.evalsTrigger = min(np.floor(10**(self.idxEvalsTrigger/self.nbptsevals)),
                                            dim * 10**self.idxDIMEvalsTrigger)
                    if self.lasteval.num < self.nbFirstEvalsToAlwaysWrite:
                        self.evalsTrigger = self.lasteval.num + 1
                    self.lasteval.is_written = True

                if ftruej - self.fopt < self.fTrigger: # minimization
                    hbuffr.append(self.lasteval.sprintData(self.fopt))
                    if ftruej <= self.fopt:
                        self.fTrigger = -np.inf
                    else:
                        if np.isinf(self.idxFTrigger):
                            self.idxFTrigger = np.ceil(np.log10(ftruej - self.fopt)) * self.nbptsf
                        while ftruej - self.fopt <= 10**(self.idxFTrigger/self.nbptsf):
                            self.idxFTrigger -= 1
                        self.fTrigger = min(self.fTrigger, 10**(self.idxFTrigger/self.nbptsf)) # TODO: why?
            # write
            if buffr:
                f = open(self.datafile, 'a')
                f.writelines(buffr)
                f.close()
            if hbuffr:
                f = open(self.hdatafile, 'a')
                f.writelines(hbuffr)
                f.close()
        else:
            self.lasteval.update(fvalue, ftrue, x)

        return fvalue

    def finalizerun(self):
        """Write the last bit of information for a given run.
        
        Calling this method at the end of a run is necessary as some
        information are not written otherwise.
        
        """
        if self._is_finalized:
            warnings.warn('Run was never started.')
            return
        if not self.lasteval.is_written:
            if not os.path.exists(self.datafile):
                warnings.warn('The data file %s is not found. '
                              'Data will be appended to an empty file. Previously '
                              'obtained data may be missing.' % self.datafile)
            f = open(self.datafile, 'a')
            f.write(self.lasteval.sprintData(self.fopt))
            f.close()

        # write in self.indexfile
        if not os.path.exists(self.indexfile):
            warnings.warn('The index file %s is not found. '
                          'Data will be appended to an empty file. Previously '
                          'obtained data may be missing.' % self.indexfile)
        f = open(self.indexfile, 'a')
        f.write(':%d|%.1e' % (self.lasteval.num,
                              self.lasteval.bestf - self.fopt - self.precision))
        f.close()
        self._is_finalized = True

    # GETTERS/SETTERS
    def _getdatapath(self):
        """Main path for storing the data."""
        return self._datapath

    def _setdatapath(self, datapath):
        try:
            os.makedirs(datapath)
        except AttributeError:
            print >>sys.stderr, 'Input argument datapath is an invalid datapath.'
            raise
        except OSError as (err, strerror):
            if err == errno.EEXIST:
                pass
            else:
                print errno, strerror
        self._datapath = datapath

    datapath = property(_getdatapath, _setdatapath)

    def _getalgid(self):
        """String representing the tested algorithm."""
        return self._algid

    def _setalgid(self, algid):
        self._algid = str(algid)

    algid = property(_getalgid, _setalgid)

    def _getcomments(self):
        """String with any information on the experiment set.

        It is recommended to put all of the parameter settings here.
        """
        return self._comments

    def _setcomments(self, comments):
        self._comments = str(comments)

    comments = property(_getcomments, _setcomments)

    def _getinputformat(self):
        """String giving the input format, either 'row' (default) or 'col'."""
        return self._inputformat

    def _setinputformat(self, inputformat):
        if inputformat in ('row', 'col'):
            self._inputformat = inputformat
        else:
            warnings.warn('The inputFormat input argument is expected to'
                          + ' match either \'col\' or \'row\'. Attempting to'
                          + 'use default (\'row\').')
            self._inputformat = 'row'

        self._is_rowformat = (self._inputformat == 'row')

    inputformat = property(_getinputformat, _setinputformat)

    def _getfbest(self):
        """Returns the best function value obtained."""

        return self.lasteval.bestf

    fbest = property(_getfbest)
    best = property(_getfbest)

    def _getftarget(self):
        """Returns the target function value."""

        try:
            return self.fopt + self.precision
        except AttributeError:
            raise # TODO

    ftarget = property(_getftarget)

    def _getevaluations(self):
        """Number of function evaluations so far."""
        if not hasattr(self, 'lasteval') or not self._is_setdim:
            # when self._is_setfun and not self._is_setdim, 
            # self.lasteval.num might not be 0. TODO: prevent this inconsistency.
            return 0. # should be synchronized with what is assigned to __Eval.num
        else:
            return self.lasteval.num

    evaluations = property(_getevaluations)

    def _getfun(self):
        """Current test function to be evaluated."""
        return self._fun

    def setfun(self, fun, fopt=None, funId='undefined', iinstance='undefined',
               dftarget=deltaftarget, **kwargs):
        """Set test function, returns an evaluation method and a target

        Using this method is necessary to start the creation of a log of
        an experiment.

        :param fun: test function (could be the evaluation method or an
                    instance of bbobbenchmarks.)
                    has to return a tuple of length 2: a noisy and a
                    noiseless value. If the function is noiseless, you
                    can provide (lambda x: tuple([f(x)] * 2)).
        :param float fopt: optimum value of the test function, used to define 
                           the f-values to be recorded  
        :param float dftarget: target precision, only used to set the ftarget 
            property attribute (must only be used for checking for final termination) 
        :param funId: function identifier, typically a number, funId must have 
                    a string representation and is used for file names (avoid
                    special characters)
        :param iinstance: instance of the function (has to have string
                          representation)
        :param kwargs: additional descriptions

        :returns: ``self``, ``self`` can be called as an objective function, 
                the target function value is self.ftarget = fopt + dftarget

        """
        if not self._is_finalized:
            self.finalizerun()

        self._is_setdim = False
        
        if fopt is None:
            fopt = fun.fopt
        
        if funId == 'undefined':
            try:
                self.funId = fun.funId
            except AttributeError:
                self.funId = funId # TODO
        else:
            self.funId = funId

        if iinstance == 'undefined':
            try:
                self.iinstance = fun.iinstance
            except AttributeError:
                self.iinstance = iinstance
        else:
            self.iinstance = iinstance

        if hasattr(self, 'fun'):
            self._is_samefun = (self.fun == fun)

        self._fun = fun
        if hasattr(fun, '_evalfull') and callable(fun._evalfull):
            self._fun_evalfull = fun._evalfull
        elif callable(fun):
            self._fun_evalfull = fun
        self.precision = dftarget
        self.fopt = fopt 
        ftarget = fopt + dftarget
        self.datafileprefix = os.path.join(self.datapath,
                                           'data_f%s' % str(self.funId),
                                           self.fileprefix)
        self.indexfile = os.path.join(self.datapath,
                                      '%s_f%s.info' % (self.fileprefix,
                                                      str(self.funId)))
        self._is_setfun = True
        self._fun_kwargs = kwargs
        # TODO: deal with *args and **kwargs... print everything out!

        # return self.evalfun, ftarget
        return self

    fun = property(_getfun, setfun)

    def _setdim(self, dim):
        """Sets the dimension.

        _setdim and setfun are fundamentally different:
        setfun is called by the user, whereas _setdim is called at the
        first call to method evalfun.

        """
        if hasattr(self, '_dim'):
            self._is_samedim = (self._dim == dim)
            if not self._is_samedim and not self._is_finalized:
                self.finalizerun()
        self._is_setdim = True
        self._dim = dim
        self.lasteval = self.__Eval()
        self.idxEvalsTrigger = 0.
        self.evalsTrigger = 1.
        self.nbFirstEvalsToAlwaysWrite = eval(nb_evaluations_always_written.replace('dim', str(dim)))  # TODO: 
        self.idxDIMEvalsTrigger = 0.
        self.fTrigger = np.inf
        self.idxFTrigger = np.inf

    def _getfileprefix(self):
        """String prefix common to all files generated."""
        return self._fileprefix

    def _setfileprefix(self, fileprefix):
        self._fileprefix = str(fileprefix) # needs to be a string

    def restart(self, restart_reason="restarted"):
        """Adds an output line to the restart-log. Call this if restarts occur within run_(your)_optimizer."""
        if self._getevaluations > 0:
            buffr = []
            buffr.append(self.lasteval.sprintData(self.fopt))
            buffr.append("% restart: "+restart_reason+"\n")
            if buffr:
                fr = open(self.rdatafile, 'a')
                fr.writelines(buffr)
                fr.close()


    fileprefix = property(_getfileprefix, _setfileprefix)

if __name__ == "__main__":
    print '  only one doctest implemented'
    # NotImplementedError('no doctests implemented')
    import doctest
    doctest.testmod()  # run all doctests in this module
    print '  done'
    
    