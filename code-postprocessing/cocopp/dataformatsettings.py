from __future__ import absolute_import, division, print_function
import numpy as np
from . import genericsettings

current_data_format = None  # used in readalign as global var


class DataFormat(object):
    """serves to define and manage the interfacing between written logger
    data and the contents of `DataSet`.

    """

    def align_data(self, aligner, data):
        """aligner is a function taking as input `data` and two column
        indices, namely where to find evaluations and function values.

        Like this we prevent to import `readalign` here, which would
        give circular imports.

        Details: this function is never used, `pproc.DataSet.__init__`
        uses `align_data_into_evals` instead.
        """
        return aligner(data, self.evaluation_idx, self.function_value_idx)

    def align_data_into_evals(self, aligner, data, dataset):
        """transfer a "raw `data` array" into the evals attribute of `dataset`.

        The raw `data` array contains for each run/trial a singlereader
        array with the data as written in the files by the logger.

        `aligner` is a function like `readalign.align_data`, taking as
        input `data` and two column indices, namely where to find
        evaluations and function values.
        """
        dataset._evals, maxevals, finalfunvals = aligner(data,
                            self.evaluation_idx, self.function_value_idx)
        assert all(dataset.evals[0][1:] == 1), dataset._evals[0]
        return maxevals, finalfunvals


class BBOBOldDataFormat(DataFormat):
    def __init__(self):
        self.evaluation_idx = 0  # index of the column where to find the evaluations
        # column idx 1 is current noise-free fitness - Fopt
        self.function_value_idx = 2  # index of the column where to find the function values (best fitness)


class BBOBNewDataFormat(DataFormat):
    """the new data format assumes constraints evaluations as second column
    """
    def __init__(self):
        self.evaluation_idx = 0  # index of the column where to find the evaluations
        self.evaluation_constraints_idx = 1  # column index for number of constraints evaluations
        self.function_value_idx = 2  # index of the column where to find the function values

    def align_data_into_evals(self, aligner, data, dataset):
        "" + DataFormat.align_data_into_evals.__doc__ + """

        Writes attributes of `dataset`, in particular `evals_constraints`,
        and `evals_function`, and `evals` as weighted sum of the two,
        unless no single constraints evaluation is found.
        """
        dataset.evals_function, maxevals, finalfunvals = aligner(data,
                                                            self.evaluation_idx,
                                                            self.function_value_idx)
        assert all(dataset.evals_function[0][1:] == 1)
        dataset.evals_constraints, maxevals_cons, finalfunvals_cons = aligner(data,
                self.evaluation_constraints_idx,
                self.function_value_idx, rewind_reader=True)

        assert all(finalfunvals == finalfunvals_cons)  # evals are different
        assert len(dataset.evals_function) >= len(dataset.evals_constraints)  # can't be > !?
        # number of (non-)nan's in both data do not agree!
        # there may be no nan's in dataset.evals_constraints (not sure why)
        # assert np.sum(np.isfinite(dataset.evals_function)) == np.sum(np.isfinite(dataset.evals_constraints))

        # check whether all constraints evaluations are zero
        # we then conclude that we don't need the evals_function and
        # evals_constraints attributes
        if not dataset.evals_constraints[:, 1:].any():  # all zero, should it better depend on testbed.has_constraints?
            # if evals_function <= 1 we rather keep attributes to be on the save side for debugging?
            dataset._evals = dataset.evals_function
            del dataset.evals_function  # clean dataset namespace
            del dataset.evals_constraints
            return maxevals, finalfunvals
        else:
            # assign dataset.evals
            dataset._evals = dataset.evals_function.copy()
            if genericsettings.weight_evaluations_constraints[0] != 1:
                dataset._evals[:,1:] *= genericsettings.weight_evaluations_constraints[0]
            # (target) f-value rows are not aligned, so we need to find for
            # each evals the respective data row in evals_constraints
            j, j_max = 0, len(dataset.evals_constraints[:, 0])
            for i, eval_row in enumerate(dataset._evals):
                # find j such that target[j] < target[i] (don't rely on floats being equal, though we probably could)
                while j < j_max and dataset.evals_constraints[j, 0] + 1e-14 > eval_row[0]:
                    j += 1  # next smaller (target) f-value
                eval_row[1:] += dataset.evals_constraints[j-1, 1:] * genericsettings.weight_evaluations_constraints[1]
            # TODO: not sure this is always what we want, but it is at least consistent with dataset.evals
            return (genericsettings.weight_evaluations_constraints[0] * maxevals +
                    genericsettings.weight_evaluations_constraints[1] * maxevals_cons,
                    finalfunvals)


class BBOBBiObjDataFormat(DataFormat):
    def __init__(self):
        self.evaluation_idx = 0  # index of the column where to find the evaluations
        self.function_value_idx = 1  # index of the column where to find the function values


data_format_name_to_class_mapping = {
        None: BBOBOldDataFormat,  # the default
        'bbob': BBOBOldDataFormat,  # the name 'bbob' is probably never used and depreciated
        'bbob-old': BBOBOldDataFormat,  # probably never used
        'bbob-new': BBOBNewDataFormat,  # 2nd column has constraints evaluations
        'bbob-new2': BBOBNewDataFormat,  # 2nd column has constraints evaluations, 5th column constraints as single digits
        'bbob-biobj': BBOBBiObjDataFormat,  # 2nd column has function evaluations
}
