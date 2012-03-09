#!/usr/bin/env python
"""python bbob_pproc should run through smoothly from a system command shell. 

this test can and should become much more sophisticated 

"""

if __name__ == "__main__": 
    import os
    data_path = '../../final-submissions/2009/data/'
    os.system('python bbob_pproc/rungeneric.py ' + 
                data_path + 'BFGS')
    os.system('python bbob_pproc/rungeneric.py --omit-single' +
                data_path + 'DE-PSO' +
                data_path + 'VNS')
