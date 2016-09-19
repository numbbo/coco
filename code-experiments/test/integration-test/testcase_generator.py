import sys  # in case we want to control what to run via command line args
import fgeneric
import bbobbenchmarks
import numpy as np


def encode_problem_index(function_idx, dimension_idx, instance_idx):
    """
    Compute the problem index for the bbob suite with 15 instances and 24 functions.
    """
    return instance_idx + (function_idx * 15) + (dimension_idx * 15 * 24)


if __name__ == '__main__':
    """
    Creates a set of vectors, evaluates them on every function in the "old" bbob suite and writes the results in a file.
    To be used by the new code to test compatibility with the old code.

    To be run in the same folder as fgeneric.py and bbobbenchmarks.py, which you can get here:
    http://coco.lri.fr/downloads/download15.03/bbobexp15.03.tar.gz
    """

    argv = sys.argv[1:]  # shortcut for input arguments

    data_path = 'folder' if len(argv) < 1 else argv[0]

    dimensions = (2, 3, 5, 10, 20, 40) if len(argv) < 2 else eval(argv[1])
    function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])
    instances = range(1, 16) if len(argv) < 4 else eval(argv[3])

    opts = dict(algid='test', comments='')

    roi = 100.0                              # region of interest is set to [-roi, roi]^D
    max_dim = 40                             # the (largest) dimension of the test vectors
    num_vectors = 100                        # number of the test vectors
    file_name = "bbob2009_testcases2.txt"

    with open(file_name, 'w') as f_out:

        x_pop = roi * (2 * np.random.rand(num_vectors, max_dim) - 1)
        f_out.write('bbob\n{}\n'.format(num_vectors))
        for x in x_pop:
            f_out.write(' '.join(map(str, x)))
            f_out.write('\n')

        f = fgeneric.LoggingFunction(data_path, **opts)
        for (dim_idx, dim) in enumerate(dimensions):  # small dimensions first, for CPU reasons
            for (fun_idx, fun_id) in enumerate(function_ids):
                for (inst_idx, i) in enumerate(instances):
                    f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=i))
                    for (x_idx, x) in enumerate(x_pop):
                        y = f.evalfun(x[0:dim])
                        # The first index is set to -1 because it is not trivial to compute and is not needed in
                        # the tests
                        f_out.write('-1\t{}\t{}\t{}\n'.format(encode_problem_index(fun_idx, dim_idx, inst_idx),
                                                              x_idx,
                                                              y))
                    f.finalizerun()

        f_out.close()
