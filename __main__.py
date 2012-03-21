#!/usr/bin/env python
"""``python bbob_pproc`` tests the package bbob_pproc and should run through 
smoothly from a system command shell. It however depends on data files that 
might not be available (to be improved). 

This test can and should become much more sophisticated.  

"""

import os, sys
# import bbob_pproc as pp

def join_path(a, *p):
    path = os.path.join(a, *p)
    return path
    
if __name__ == "__main__": 
    """these tests are executed when ``python bbob_pproc`` is called.  

    with ``wine`` as second argument ``C:\\Python26\\python.exe`` 
    instead of ``python`` is called
    
    """
    python = 'python '  # how to call python 
    if len(sys.argv) > 1 and sys.argv[1] == 'wine':
        python = 'C:\\Python26\\python.exe ' # works for wine
    
    data_path = join_path(' ..', '..', 'final-submissions', '2009', 'data')

    command = join_path(' bbob_pproc', 'rungeneric.py ')
    
    print '*** testing module bbob_pproc ***'

    os.system(python + command + '--conv' + 
                join_path(data_path, 'BFGS'))
    os.system(python + command + ' --omit-single ' +
                join_path(data_path, 'DE-PSO ') +
                join_path(data_path, 'VNS '))
    os.system(python + command + ' --omit-single ' +
                join_path(data_path, 'BIPOP-CMA-ES ') +
                join_path(data_path, 'PSO ') +
                join_path(data_path, 'ALPS '))

    print '*** done testing module bbob_pproc ***'
