#!/usr/bin/env python
"""python bbob_pproc should run through smoothly from a system command shell. 

this test can and should become much more sophisticated 

"""

if __name__ == "__main__": 
    print '*** testing module bbob_pproc ***'

    import os
    data_path = os.path.join(' ..', '..', 'final-submissions', '2009', 'data')
    
    os.system('python bbob_pproc/rungeneric.py ' + 
                os.path.join(data_path, 'BFGS'))
    os.system('python bbob_pproc/rungeneric.py --omit-single ' +
                os.path.join(data_path, 'DE-PSO ') +
                os.path.join(data_path, 'VNS '))
    os.system('python bbob_pproc/rungeneric.py --omit-single ' +
                os.path.join(data_path, 'BIPOP-CMA-ES ') +
                os.path.join(data_path, 'PSO ') +
                os.path.join(data_path, 'ALPS '))

    print '*** done testing module bbob_pproc ***'
