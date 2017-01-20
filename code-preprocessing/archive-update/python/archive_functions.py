from __future__ import absolute_import, division, print_function, unicode_literals

import os

from .archive_exceptions import PreprocessingWarning, PreprocessingException
from .archive_load_data import create_path, get_key_value, get_file_name_list, get_archive_file_info, get_range


class ProblemInstanceInfo:
    """Contains information on the problem instance: suite_name, function, instance, dimension and a list of file names
       with archived solutions for this problem instance.
    """

    def __init__(self, _file_name, suite_name, function, instance, dimension):
        """Instantiates a ProblemInstanceInfo object.
        """
        self.suite_name = suite_name
        self.function = function
        self.instance = instance
        self.dimension = dimension
        self.file_names = [_file_name]

        self.current_file_initialized = False
        self.current_position = 0
        self.current_file_index = 0

    def __str__(self):
        return "{}_f{:02d}_i{:02d}_d{:02d}".format(self.suite_name, self.function, self.instance, self.dimension)

    def equals(self, suite_name, function, instance, dimension):
        """Returns true if this self has the same suite_name, function, instance and dimension as the given ones and
           false otherwise.
           :param suite_name: suite name
           :param function: function number
           :param instance: instance number
           :param dimension: dimension
        """
        if self.suite_name != suite_name:
            return False
        if self.function != function:
            return False
        if self.instance != instance:
            return False
        if self.dimension != dimension:
            return False
        return True

    def fill_archive(self, archive):
        """Reads the solutions from the files and feeds them to the given archive.
           :param archive: archive to be filled with solutions
        """
        for f_name in self.file_names:
            #print(f_name)
            with open(f_name, 'r') as f:

                instance_found = False

                for line in f:
                    if not line.strip() or (line[0] == '%' and 'instance' not in line):
                        # Ignore empty lines
                        continue

                    elif line[0] == '%' and 'instance' in line:
                        if instance_found:
                            break
                        else:
                            value = get_key_value(line[1:], 'instance')
                            if int(value) == self.instance:
                                instance_found = True

                    elif instance_found:
                        if line[0] != '%':
                            # Solution found, feed it to the archive
                            try:
                                archive.add_solution(float(line.split()[1]), float(line.split()[2]), line)
                            except IndexError:
                                print('Problem in file {}, line {}, skipping line'.format(f_name, line))
                                continue

                f.close()
                if not instance_found:
                    raise PreprocessingException('File \'{}\' does not contain \'instance = {}\''.format(f_name,
                                                                                                         self.instance))

    # noinspection PyTypeChecker
    def write_archive_solutions(self, output_path, archive, crop_variables):
        """Appends solutions to a file in the output_path named according to self's suite_name, function, instance and
           dimension.
           :param archive: archive to be output
           :param output_path: output path (if it does not yet exist, it is created)
           :param crop_variables: if true, the variables are omitted from the output
        """
        create_path(output_path)
        f_name = os.path.join(output_path, '{}_f{:02d}_i{:02d}_d{:02d}_nondominated.adat'.format(self.suite_name,
                                                                                                 self.function,
                                                                                                 self.instance,
                                                                                                 self.dimension))
        with open(f_name, 'a') as f:
            f.write('% instance = {}\n%\n'.format(self.instance))

            while True:
                text = archive.get_next_solution_text()
                if text is None:
                    break
                if crop_variables:
                    f.write('\t'.join(text.split()[:3]) + '\n')
                else:
                    f.write(text)

            f.close()


class ArchiveInfo:
    """Collects information on the problem instances contained in all archives.
    """

    def __init__(self, input_paths, functions, instances, dimensions, output_files=True):
        """Instantiates an ArchiveInfo object.
           Extracts information from all archives found in the input_paths that correspond to the given functions,
           instances and dimensions. Returns the resulting ArchiveInfo.
        """

        self.problem_instances = []
        self.current_instance = 0
        count = 0

        # Read the information on the archive
        input_files = get_file_name_list(input_paths, ".adat")
        if len(input_files) == 0:
            raise PreprocessingException('Folder \'{}\' does not exist or is empty'.format(input_paths))

        archive_info_list = []
        for input_file in input_files:
            try:
                archive_info_set = get_archive_file_info(input_file, functions, instances, dimensions)
                if archive_info_set is not None and len(archive_info_set) > 0:
                    archive_info_list.append(archive_info_set)
                    count += 1
                    if output_files:
                        print(input_file)

            # If any problems are encountered, the file is skipped
            except PreprocessingWarning as warning:
                print(warning)

        print('Successfully processed archive information from {} files.'.format(count))

        # Store archive information only for instances that correspond to instance_list
        print('Storing archive information...')
        for archive_info_set in archive_info_list:
            for archive_info_entry in archive_info_set:
                self._add_entry(*archive_info_entry)

    def __str__(self):
        result = ""
        for problem_instance in self.problem_instances:
            if problem_instance is not None:
                result += str(problem_instance) + '\n'
        return result

    def _add_entry(self, _file_name, suite_name, function, instance, dimension):
        """Adds a new ProblemInstanceInfo instance with the given suite_name, function, instance, dimension to the list
           of problem instances if an instance with these exact values does not exist yet. If it already exists, the
           current file_name is added to its list of file names.
        """

        found = False
        for problem_instance in self.problem_instances:
            if problem_instance.equals(suite_name, function, instance, dimension):
                problem_instance.file_names.append(_file_name)
                found = True
                break

        if not found:
            self.problem_instances.append(ProblemInstanceInfo(_file_name, suite_name, function, instance, dimension))

    def get_next_problem_instance_info(self):
        """Returns the current ProblemInstanceInfo and increases the counter. If there are no more instances left,
           returns None.
        """
        if self.current_instance >= len(self.problem_instances):
            return None

        self.current_instance += 1
        return self.problem_instances[self.current_instance - 1]

    def get_function_string(self):
        """Returns all the contained functions in a string form (using ranges where possible).
        """
        function_set = set()
        for problem_instance in self.problem_instances:
            function_set.add(problem_instance.function)
        return get_range(function_set)

    def get_instance_string(self):
        """Returns all the contained instances in a string form (using ranges where possible).
        """
        instance_set = set()
        for problem_instance in self.problem_instances:
            instance_set.add(problem_instance.instance)
        return get_range(instance_set)

    def get_dimension_string(self):
        """Returns all the contained dimensions in a string form.
        """
        dimension_set = set()
        for problem_instance in self.problem_instances:
            dimension_set.add(problem_instance.dimension)
        return ','.join(str(s) for s in sorted(dimension_set))

    def get_file_name_set(self):
        """Returns the set of all contained files.
        """
        file_name_set = set()
        for problem_instance in self.problem_instances:
            for file_name in problem_instance.file_names:
                file_name_set.add(file_name)
        return sorted(file_name_set)

