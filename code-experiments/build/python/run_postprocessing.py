from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from bbob_pproc import rungeneric

if __name__ == "__main__":
    print(sys.argv)
    rungeneric.main(sys.argv[1:])
