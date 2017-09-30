from __future__ import absolute_import, division, print_function
import numpy as np

current_data_format = None  # is currently hidden under testbedsettings.data_format, so this needs clean-up

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
        dataset.evals, maxevals, finalfunvals = aligner(data,
                            self.evaluation_idx, self.function_value_idx)
        assert all(dataset.evals[0][1:] == 1)
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

        Writes attributes of `dataset`, namely `evals_constraints`,
        `evals_function`, and `evals` as sum of the two.
        """
        # print('in align_data_into_evals')
        dataset.evals_function, maxevals, finalfunvals = aligner(data,
                                                            self.evaluation_idx,
                                                            self.function_value_idx)
        assert all(dataset.evals_function[0][1:] == 1)
        dataset.evals_constraints, maxevals_cons, finalfunvals_cons = aligner(data,
                self.evaluation_constraints_idx,
                self.function_value_idx, rewind_reader=True)

        assert all(finalfunvals == finalfunvals_cons)  # evals are different
        # number of (non-)nan's in both data must agree
        assert np.nansum(dataset.evals_function > -1) == np.nansum(dataset.evals_constraints > -1)
        assert len(dataset.evals_function) >= len(dataset.evals_constraints)

        # check whether all constraints evaluations are zero
        # we then conclude that we don't need the _function and
        # _constraints attributes
        if np.nanmax(dataset.evals_constraints) == 0 and np.nanmax(dataset.evals_function) > 1:
            # if evals_function <= 1 we rather keep attributes to be on the save side for debugging
            dataset.evals = dataset.evals_function
            del dataset.evals_function  # clean dataset namespace
        else:
            # for the time being we add evals_functions and evals_constraints
            dataset.evals = dataset.evals_function.copy()
            # (target) f-value rows are not aligned, so we need to find for
            # each evals the respective data row in evals_constraints
            j, j_max = 0, len(dataset.evals_constraints[:, 0])
            for i, eval_row in enumerate(dataset.evals):
                # find j such that target[j] < target[i] (don't rely on floats being equal, though we probably could)
                while j < j_max and dataset.evals_constraints[j, 0] + 1e-14 >= eval_row[0]:
                    j += 1  # next smaller (target) f-value
                eval_row[1:] += dataset.evals_constraints[j-1, 1:]
            # print(dataset.evals_function, dataset.evals_constraints, dataset.evals)
        return maxevals, finalfunvals


class BBOBBiObjDataFormat(DataFormat):
    def __init__(self):
        self.evaluation_idx = 0  # index of the column where to find the evaluations
        self.function_value_idx = 1  # index of the column where to find the function values

data_format_name_to_class_mapping = {
        None: BBOBOldDataFormat,  # the default
        'bbob': BBOBOldDataFormat,  # the name 'bbob' is probably never used and depreciated
        'bbob-old': BBOBOldDataFormat,  # probably never used
        'bbob-new': BBOBNewDataFormat,  # 2nd column has constraints evaluations
        'bbob-biobj': BBOBBiObjDataFormat,  # 2nd column has function evaluations
}

def get_data_format(name):
    """return the respective data format class instance.

    So far the code only works, because the data_format is assigned in the
    testbed class and fixed for each testbed class, which somehow defeats
    its original purpose.
    """
    return data_format_name_to_class_mapping[name]()

def set_data_format(name):
    """set global variable `dataformatsettings.current_data_format`.
    
    This is probably not the right way to do this.
    """
    global current_data_format
    current_data_format = get_data_format(name)

