from __future__ import absolute_import, division, print_function, unicode_literals

import os

from .archive_exceptions import PreprocessingWarning, PreprocessingException
from .archive_load_data import create_path, get_key_value, get_file_name_list, get_archive_file_info
from .archive_load_data import read_best_values, write_best_values
from .coco_archive import Archive, log_level


class ProblemInstanceInfo:
    """Contains information on the problem instance: suite_name, function, dimension, instance and a list of file names
       with archived solutions for this problem instance.
    """

    def __init__(self, _file_name, suite_name, function, dimension, instance):
        """Instantiates a ProblemInstanceInfo object.
        """
        self.suite_name = suite_name
        self.function = function
        self.dimension = dimension
        self.instance = instance
        self.file_names = [_file_name]

        self.current_file_initialized = False
        self.current_position = 0
        self.current_file_index = 0

    def __str__(self):
        return "{}_f{:02d}_i{:02d}_d{:02d}".format(self.suite_name, self.function, self.instance, self.dimension)

    def equals(self, suite_name, function, dimension, instance):
        """Returns true if this self has the same suite_name, function, dimension and instance as the given ones and
           false otherwise.
           :param instance: instance number
           :param dimension: dimension
           :param function: function number
           :param suite_name: suite name
        """
        if self.suite_name != suite_name:
            return False
        if self.function != function:
            return False
        if self.dimension != dimension:
            return False
        if self.instance != instance:
            return False
        return True

    def fill_archive(self, archive):
        """Reads the solutions from the files and feeds them to the given archive.
           :param archive: archive to be filled with solutions
        """
        for f_name in self.file_names:
            with open(f_name, 'r') as f:

                instance_found = False
                stop_reading = False

                for line in f:
                    if not line.strip():
                        # Ignore empty lines
                        continue
                    if (line[0] == '%') and stop_reading:
                        break
                    elif (line[0] == '%') and (not instance_found):
                        if 'instance' in line:
                            value = get_key_value(line[1:], 'instance')
                            if int(value) == self.instance:
                                instance_found = True
                    elif instance_found:
                        if line[0] != '%':
                            if not stop_reading:
                                stop_reading = True
                            # Solution found, feed it to the archive
                            archive.add_solution(float(line.split('\t')[1]), float(line.split('\t')[2]), line)

                f.close()
                if not instance_found:
                    raise PreprocessingException('File \'{}\' does not contain \'instance = {}\''.format(f_name,
                                                                                                         self.instance))

    # noinspection PyTypeChecker
    def write_archive_solutions(self, output_path, archive):
        """Appends solutions to a file in the output_path named according to self's suite_name, function and dimension.
           :param archive: archive to be output
           :param output_path: output path (if it does not yet exist, it is created)
        """
        create_path(output_path)
        f_name = os.path.join(output_path, '{}_f{:02d}_d{:02d}_nondominated.adat'.format(self.suite_name,
                                                                                         self.function,
                                                                                         self.dimension))
        with open(f_name, 'a') as f:
            f.write('% instance = {}\n%\n'.format(self.instance))

            while True:
                text = archive.get_next_solution_text()
                if text is None:
                    break
                f.write(text)

            f.close()


class ArchiveInfo:
    """Collects information on the problem instances contained in all archives.
    """

    def __init__(self, input_path):
        """Instantiates an ArchiveInfo object.
           Extracts information from all archives found in the input_path and returns the resulting ArchiveInfo.
        """

        self.problem_instances = []
        self.current_instance = 0
        count = 0

        # Read the information on the archive
        input_files = get_file_name_list(input_path)
        if len(input_files) == 0:
            raise PreprocessingException('Folder \'{}\' does not exist or is empty'.format(input_path))

        archive_info_list = []
        for input_file in input_files:
            try:
                archive_info_set = get_archive_file_info(input_file)
            # If any problems are encountered, the file is skipped
            except PreprocessingWarning as warning:
                print(warning)
            else:
                archive_info_list.append(archive_info_set)
                count += 1

        print('Successfully processed archive information from {} files.'.format(count))

        # Store archive information
        print('Storing archive information...')
        for archive_info_set in archive_info_list:
            for archive_info_entry in archive_info_set:
                self._add_entry(*archive_info_entry)

    def __str__(self):
        result = ""
        for instance in self.problem_instances:
            if instance is not None:
                result += str(instance) + '\n'
        return result

    def _add_entry(self, _file_name, suite_name, function, dimension, instance):
        """Adds a new ProblemInstanceInfo instance with the given suite_name, function, dimension, instance to the list
           of problem instances if an instance with these exact values does not exist yet. If it already exists, the
           current file_name is added to its list of file names.
        """
        found = False
        for problem_instance in self.problem_instances:
            if problem_instance.equals(suite_name, function, dimension, instance):
                problem_instance.file_names.append(_file_name)
                found = True
                break

        if not found:
            self.problem_instances.append(ProblemInstanceInfo(_file_name, suite_name, function, dimension, instance))

    def get_next_problem_instance_info(self):
        """Returns the current ProblemInstanceInfo and increases the counter. If there are no more instances left,
           returns None.
        """
        if self.current_instance >= len(self.problem_instances):
            return None

        self.current_instance += 1
        return self.problem_instances[self.current_instance - 1]


def update_best_hypervolume(old_best_files, new_best_data, new_best_file):
    """Updates the best hypervolume values. The old hypervolume values are read from old_best_files (a list of files),
       while the new ones are passed through new_best_data. The resulting best values are appended to new_best_file
       in a format that can be readily used by the COCO source code in C.
       :param old_best_files: list of files containing best hypervolumes
       :param new_best_data: dictionary with problem names and their new best hypervolumes
       :param new_best_file: name of the file to which the new values will be appended
    """
    print('Updating best hypervolume values...')

    # Read the old best values from the given files
    try:
        old_best_data = read_best_values(old_best_files)
    except IOError as err:
        print(err)
        print('Continuing nevertheless...')
        result = new_best_data
    else:
        # Create a set of problem_names contained in at least one dictionary
        problem_names = set(old_best_data.keys()).union(set(new_best_data.keys()))
        result = {}

        # Iterate over all problem names and store only the best hypervolumes
        for problem_name in problem_names:
            new_value = new_best_data.get(problem_name)
            old_value = old_best_data.get(problem_name)
            if new_value is None:
                result.update({problem_name: float(old_value)})
            elif old_value is None:
                result.update({problem_name: float(new_value)})
            else:
                result.update({problem_name: min(float(new_value), float(old_value))})

    # Write the best values
    write_best_values(result, new_best_file)
    print('Done.')


def merge_archives(input_path, output_path):
    """Merges all archives from the input_path (removes any dominated solutions) and stores the consolidated archives
       in the output_path. Returns problem names and their new best hypervolume values in the form of a dictionary.
       :param input_path: input path
       :param output_path: output path (created if not existing before)
    """
    result = {}

    print('Reading archive information...')
    archive_info = ArchiveInfo(input_path)

    print('Processing archives...')
    while True:
        # Get information about the next problem instance
        problem_instance_info = archive_info.get_next_problem_instance_info()
        if problem_instance_info is None:
            break
        print(problem_instance_info)

        old_level = log_level('warning')

        # Create an archive for this problem instance
        archive = Archive(problem_instance_info.suite_name, problem_instance_info.function,
                          problem_instance_info.dimension, problem_instance_info.instance)

        # Read the solutions from the files and add them to the archive
        problem_instance_info.fill_archive(archive)

        # Write the non-dominated solutions into output folder
        problem_instance_info.write_archive_solutions(output_path, archive)

        result.update({str(problem_instance_info): archive.hypervolume})

        log_level(old_level)

    return result
