"""import this before numpy is imported to fix hangups with 100% CPU usage on Linux.

    see https://github.com/numbbo/coco/issues/1919
    and https://github.com/CMA-ES/pycma/issues/238
    and https://twitter.com/jeremyphoward/status/1185044752753815552
"""

import sys

### MKL bug fix
def set_num_threads(nt=1, disp=1):
    """see https://github.com/numbbo/coco/issues/1919
    and https://github.com/CMA-ES/pycma/issues/238
    and https://twitter.com/jeremyphoward/status/1185044752753815552
    """
    import os
    try: import mkl
    except ImportError: disp and print("mkl is not installed")
    else:
        mkl.set_num_threads(nt)
    nt = str(nt)
    for name in ['OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS',
                 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS']:
        os.environ[name] = nt
    disp and print("setting openblas, numexpr, omp and mkl threads num to", nt)

if sys.platform.lower() not in ('darwin', 'windows'):
    set_num_threads(1)  # execute before numpy is imported
