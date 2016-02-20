# -*- coding: utf-8 -*-
"""Methods for merging the archives.
   
   Input archives are read and merged so that only non-dominated solutions are
   stored in the output archives. A file with the best known hypervolume values
   is generated from these hypervolumes and the ones stored in C source files. 
"""
import os
import pyximport; pyximport.install()

import load_data as ld
from archive_info import ArchiveInfo
from archive import Archive

#import mock_archive as a
# TODO replace mock archive with the real deal
    
def update_best_hypervolume(old_best_files, new_best_data, new_best_file):
    """Updates the best hypervolume values. The old hypervolume values are read
       from old_best_files (a list of files), while the new ones are passed 
       through new_best_data. The resulting best values are appended to new_best_file 
       in a format that can be readily used by the COCO source code in C.
    """
    print("Updating best hypervolume values...")
    
    # Read the old best values from the given files
    try:
        old_best_data = ld.read_best_values(old_best_files)
    except FileNotFoundError as err:
        print(err)
        print("Continuing nevertheless...")
        result = new_best_data        
    else:
        # Create a set of problem_names contained in at least one dictionary
        problem_names = set(old_best_data.keys()).union(set(new_best_data.keys()))     
        result = {}    
        
        # Iterate over all problem names and store only the best hypervolumes
        for problem_name in problem_names:
            new_value = new_best_data.get(problem_name)
            old_value = old_best_data.get(problem_name)
            if (new_value is None):
                result.update({problem_name : float(old_value)})
            elif (old_value is None):
                result.update({problem_name : float(new_value)})
            else:
                result.update({problem_name : min(float(new_value), float(old_value))})
                
    # Write the best values
    ld.write_best_values(result, new_best_file)
    print("Done.")

def merge_archives(input_path, output_path):
    """Merges all archives from the input_path (removes any dominated solutions) 
       and stores the consolidated archives in the output_path. Returns problem 
       names and their new best hypervolume values in the form of a dictionary.
    """
    result = {}    
    
    print("Reading archive information...")
    archive_info = ArchiveInfo(input_path)
    
    print("Processing archives...")
    while True:
        # Get information about the next problem instance
        problem_instance_info = archive_info.get_next_problem_instance_info()
        if problem_instance_info is None:
            break
        print(problem_instance_info)
        
        # Create an archive for this problem instance
        archive = Archive(problem_instance_info.suite_name,
                          problem_instance_info.function,
                          problem_instance_info.dimension,
                          problem_instance_info.instance)
        
        # Read the solutions from the files and add them to archive
        problem_instance_info.fill_archive(archive)
        
        # Write the non-dominated solutions into output folder
        problem_instance_info.write_archive_solutions(output_path, archive)        
        
        result.update({str(problem_instance_info) : archive.hypervolume})   
    
    return result
        
if __name__ == '__main__':
    basepath = os.path.dirname(__file__)
    archive_input = os.path.abspath(os.path.join(basepath, "archives-input"))
    archive_output = os.path.abspath(os.path.join(basepath, "archives-output"))
    new_hypervolumes = merge_archives(archive_input, archive_output)
    
    # Use files with best hypervolume values from the src folder:
    file_names = ['suite_biobj_best_values_hyp.c']
    file_names = [os.path.abspath(os.path.join(basepath, "..", "..", 
    "code-experiments/src", file_name)) for file_name in file_names]
    
    update_best_hypervolume(file_names, new_hypervolumes, 'new_best_values_hyp.c')
