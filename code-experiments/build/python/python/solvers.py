from __future__ import absolute_import, division, print_function
import numpy as np

# ===============================================
# the most basic example solver
# ===============================================
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between
    `lbounds` and `ubounds`
    """
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), None, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(max([1, min([budget, max_chunk_size])]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
        if fun.number_of_constraints > 0:
            CF = [[fun.constraint(x), fun(x)] for x in X]  # call f and constraint at the "same" time
            F = [cf[1] for cf in CF]  # for indexing argmin
        else:
            F = [fun(x) for x in X]
        if fun.number_of_objectives == 1:
            if fun.number_of_constraints > 0:
                idx_feasible = np.where([np.all(cf[0] <= 0) for cf in CF])[0]
                index = idx_feasible[np.argmin(np.asarray(F)[idx_feasible])] if len(idx_feasible) else None
            else:
                index = np.argmin(F)
            if index is not None and (f_min is None or F[index] < f_min):
                x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min


