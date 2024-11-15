import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from archive_load_data import get_file_name_list, create_path, parse_archive_file_name #might need to be changed to 'from cocoprep.archive_load_data', but the cocoprep module could not be located so we adapted it to this.

def read_adat_file(file_path):
    """
    Reads a .adat file and returns the function evaluations and non-dominated points.
    
    Parameters:
    file_path (str): Path to the .adat file.
    
    Returns:
    tuple: (function_evals, archive_sizes), where:
        - function_evals (list): Cumulative function evaluations at each logging point.
        - archive_sizes (list): Archive size (number of non-dominated points) at each logging point.
    """
    function_evals = []
    archive_sizes = []
    archive_size = 0  # Track the number of non-dominated points
    
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('%'):
                data = line.strip().split()
                eval_count = int(data[0])  # Function evaluations
                archive_size += 1  # Increment for each line, as each line represents a new non-dominated point
                function_evals.append(eval_count)
                archive_sizes.append(archive_size)
    
    return function_evals, archive_sizes

def plot_archive_size_by_dimension(output_dir="plots"):
    """
    Generates separate plots of archive size over time for each dimension,
    combining all functions and instances for that dimension.
    
    Parameters:
    output_dir (str): Directory to save the plots.
    """
    create_path(output_dir)
    
    # Collect all .adat files in the current directory
    file_paths = get_file_name_list(".", ending=".adat")
    
    # Data storage by dimension
    data_by_dimension = defaultdict(lambda: defaultdict(list))
    
    # Read data from each file
    for file_path in file_paths:
        try:
            suite_name, function, instance, dimension = parse_archive_file_name(file_path)
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")
            continue
        function_evals, archive_sizes = read_adat_file(file_path)
        
        # Store data by dimension and function
        data_by_dimension[dimension][function].append((function_evals, archive_sizes))

    # Generate a plot for each dimension
    for dimension, functions in data_by_dimension.items():
        plt.figure(figsize=(12, 8))
        plt.title(f"Archive Size vs Function Evaluations for Dimension {dimension}")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Archive Size")
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(functions)))
        
        for i, (function_name, instances) in enumerate(functions.items()):
            color = colors[i]
            for evals, sizes in instances:
                plt.plot(evals, sizes, color=color, linewidth=0.7, alpha=0.5)
            
            plt.plot([], [], color=color, label=f"{function_name} instances")
        
        plt.legend(title="Function", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small", framealpha=0.5)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        plot_filename = f"archive_size_{dimension}.png"
        plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches="tight")
        plt.close()
        print(f"Plot for dimension {dimension} saved as {plot_filename}")

def plot_average_archive_size_by_dimension(output_dir="plots"):
    """
    Generates separate plots of average archive size over time for each dimension,
    combining all functions and averaging over instances for that function.
    
    Parameters:
    output_dir (str): Directory to save the plots.
    """
    create_path(output_dir)
    
    # Collect all .adat files in the current directory
    file_paths = get_file_name_list(".", ending=".adat")
    
    # Data storage by dimension
    data_by_dimension = defaultdict(lambda: defaultdict(list))
    
    # Read data from each file
    for file_path in file_paths:
        try:
            suite_name, function, instance, dimension = parse_archive_file_name(file_path)
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")
            continue
        function_evals, archive_sizes = read_adat_file(file_path)
        
        # Store data by dimension and function
        data_by_dimension[dimension][function].append((function_evals, archive_sizes))

    # Generate a plot for each dimension
    for dimension, functions in data_by_dimension.items():
        plt.figure(figsize=(12, 8))
        plt.title(f"Average Archive Size vs Function Evaluations for Dimension {dimension}")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Average Archive Size")
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(functions)))
        
        for i, (function_name, instances) in enumerate(functions.items()):
            color = colors[i]
            
            max_eval_count = max(max(evals) for evals, _ in instances)
            archive_sizes_sum = np.zeros(max_eval_count + 1)
            count_per_eval = np.zeros(max_eval_count + 1)
            
            for evals, sizes in instances:
                current_archive_size = 0
                eval_index = 0
                
                for eval_count in range(1, max_eval_count + 1):
                    if eval_index < len(evals) and evals[eval_index] == eval_count:
                        current_archive_size = sizes[eval_index]
                        eval_index += 1
                    
                    archive_sizes_sum[eval_count] += current_archive_size
                    count_per_eval[eval_count] += 1
            
            with np.errstate(divide='ignore', invalid='ignore'):
                average_archive_size = np.divide(archive_sizes_sum, count_per_eval, where=count_per_eval > 0)
                average_archive_size = np.nan_to_num(average_archive_size)
            
            eval_points = np.arange(1, len(average_archive_size))
            
            plt.plot(eval_points, average_archive_size[1:], color=color, label=function_name)
        
        plt.legend(title="Function", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small", framealpha=0.5)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        plot_filename = f"average_archive_size_{dimension}.png"
        plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches="tight")
        plt.close()
        print(f"Average plot for dimension {dimension} saved as {plot_filename}")

if __name__ == "__main__":
    plot_archive_size_by_dimension()
    plot_average_archive_size_by_dimension()
