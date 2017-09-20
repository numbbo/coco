class DataFormat(object):

    settings = dict()

    def __init__(self):
        for key, val in self.settings.items():
            setattr(self, key, val)


class BBOBDataFormat(DataFormat):

    settings = dict(
        evaluation_idx=0,  # index of the column where to find the evaluations
        function_value_idx=2  # index of the column where to find the function values
    )


class BiObjBBOBDataFormat(DataFormat):

    settings = dict(
        evaluation_idx=0,  # index of the column where to find the evaluations
        function_value_idx=1  # index of the column where to find the function values
    )


data_format_translation = {
    'bbob': BBOBDataFormat(),
    'bbob-biobj': BiObjBBOBDataFormat(),
}
