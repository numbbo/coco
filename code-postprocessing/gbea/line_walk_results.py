from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import ceil, isclose
import warnings


def reformat_line_walk_results(path):
    """Performs reformatting of the line walk results output from the rw COCO logger in all files
    in path to ease plotting.
    """
    for in_file_name in os.listdir(path):
        if '_rw' not in in_file_name:
            continue
        out_file_name = os.path.join(path, in_file_name.replace('_rw', '_reformatted'))
        time_file_name = os.path.join(path, in_file_name.replace('_rw', '_time'))
        with open(os.path.join(path, in_file_name), 'r') as f_in, \
                open(out_file_name, 'w') as f_out:
            end_of_comments = False
            f_time = None
            previous_line = None
            previous_step = None
            previous_x = 0
            current_x = 0
            current_step = None
            for line in f_in:
                split_line = line.split()
                if line[0] == '%' or line.strip() == '':
                    if "variables" in split_line:
                        dim = int(split_line[split_line.index("variables") - 1])
                        end_of_comments = True
                    if "time" in split_line:
                        f_time = open(time_file_name, 'w')
                else:
                    numbers = split_line[:dim+2]
                    if f_time:
                        f_time.write("{}\t{}\n".format(numbers[1], split_line[dim+2]))
                    if end_of_comments:
                        # The line containing the origin point
                        dim = len(numbers) - 2
                        f_out.write("# origin\n")
                        for n in numbers[2:]:
                            f_out.write("{}\t{}\t{}\t\n".format(numbers[0], numbers[1], n))
                        end_of_comments = False
                    elif previous_line is None:
                        # First line of a new sequence
                        previous_line = [n for n in numbers]
                        f_out.write("\n# x{}\n".format(current_x))
                        f_out.write("{}\t{}\t{}\t\n".format(numbers[0], numbers[1],
                                                            numbers[current_x + 2]))
                    elif previous_step is None:
                        # Second line of a new sequence
                        for i, (c, p) in enumerate(zip(numbers[2:], previous_line[2:])):
                            if c != p:
                                current_step = float(c) - float(p)
                                current_x = i
                                break
                        f_out.write("{}\t{}\t{}\t\n".format(numbers[0], numbers[1],
                                                            numbers[current_x + 2]))
                        previous_line = [n for n in numbers]
                        previous_step = current_step
                        previous_x = current_x
                    else:
                        # Next line
                        current_step = 0
                        for i, (n, p) in enumerate(zip(numbers[2:], previous_line[2:])):
                            if n != p:
                                current_step = float(n) - float(p)
                                current_x = i
                        if isclose(current_step, previous_step) and current_x == previous_x:
                            # Same sequence
                            f_out.write("{}\t{}\t{}\t\n".format(numbers[0], numbers[1],
                                                                numbers[current_x + 2]))
                        else:
                            # New sequence
                            if current_x == previous_x:
                                current_x += 1
                                if current_x == dim:
                                    break
                            f_out.write("\n# x{}\n".format(current_x))
                            f_out.write("{}\t{}\t{}\t\n".format(numbers[0], numbers[1],
                                                                numbers[current_x + 2]))
                            previous_step = None
                        previous_line = [n for n in numbers]
                        previous_x = current_x
            if f_time:
                f_time.close()


def load_line_walk_results(file_name):
    """Returns the line walk results stored in file_name as a dictionary. """
    array = None
    current_set = None
    result_list = {}
    try:
        with open(file_name, 'r') as f:
            for line in f:
                if line[0] == '#':
                    current_set = line.split()[1]
                elif line.strip() == '':
                    # The array is completed, store it in the list
                    result_list[current_set] = array
                    array = None
                elif array is None:
                    array = np.array(line.split(), dtype=float)
                else:
                    line_array = np.array(line.split(), dtype=float)
                    array = np.row_stack((array, line_array))
    except FileNotFoundError:
        warnings.warn("Skipping file {}".format(file_name))
    # Take care of the last round
    result_list[current_set] = array
    return result_list


def plot_line_walk_results(file_names, pdf=None, title=None, n_rows=4, n_columns=8):
    """Plots the line walk results stored in file_names. """
    results = []
    for file_name in file_names:
        results.append(load_line_walk_results(file_name))
    dim = len(results[0]) - 1
    for batch in range(int(ceil(dim / n_columns / n_rows))):
        fig, _ = plt.subplots(n_rows, n_columns, figsize=(2 * n_columns, 2 * n_rows))
        for i in range(n_rows * n_columns):
            x = batch * n_rows * n_columns + i
            for j, result in enumerate(results):
                try:
                    array = result["x{}".format(x)]
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][j % 10]
                    fig.axes[i].plot(array[:, 2], array[:, 1], color=color)
                    origin = result["origin".format(x)]
                    fig.axes[i].plot(origin[x, 2], origin[x, 1], 'o', markersize=4, color=color,
                                     markeredgecolor='black', markeredgewidth=0.5)
                except KeyError:
                    continue
            fig.axes[i].set_title("x{}".format(x))
            if x >= dim:
                fig.axes[i].set_visible(False)
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        if pdf is not None:
            pdf.savefig(fig)
        else:
            plt.show()
        plt.close()


def load_line_walk_times(file_names):
    """Returns the line walk times stored in file_names as a list of dictionaries. """
    time_list = []
    for file_name in sorted(file_names):
        parts = file_name.split('_')
        f = int(parts[1][1:])
        i = int(parts[2][1:])
        try:
            times = np.loadtxt(file_name)
            f_exists = False
            for tl in time_list:
                if f == tl['f']:
                    f_exists = True
                    tl['instances'].append({'i': i, 'times': times})
                    break
            if not f_exists:
                time_list.append({'f': f, 'instances': [{'i': i, 'times': times}]})
        except IOError:
            warnings.warn("Skipping file {}".format(file_name))
    return time_list


def plot_line_walk_times(file_names, pdf=None, title=None, n_rows=2, n_columns=3):
    """Plots the line walk times stored in file_names. """
    time_list = load_line_walk_times(file_names)
    num = len(time_list)
    for batch in range(int(ceil(num / n_columns / n_rows))):
        fig, _ = plt.subplots(n_rows, n_columns, figsize=(2.5 * n_columns, 2 * n_rows))
        for j in range(n_rows * n_columns):
            x = batch * n_rows * n_columns + j
            if x >= num:
                fig.axes[j].set_visible(False)
                continue
            f = time_list[x]['f']
            instances = time_list[x]['instances']
            for instance in instances:
                i = instance['i']
                times = instance['times']
                color = plt.rcParams['axes.prop_cycle'].by_key()['color'][(i - 1) % 10]
                fig.axes[j].plot(times[:, 1], times[:, 0], 'o', markersize=4, color=color, alpha=0.5)
            fig.axes[j].set_xlabel("time (s)")
            fig.axes[j].set_ylabel("f{}".format(f))
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if pdf is not None:
            pdf.savefig(fig)
        else:
            plt.show()
        plt.close()


def rw_gan_mario_line_walk(exdata_path, point='random'):
    suite_name = "rw-gan-mario"
    path = os.path.join(exdata_path, "{}-line-walk-{}".format(suite_name, point))
    reformat_line_walk_results(path)
    pdf_file_name = os.path.join(path, "{}-{}.pdf".format(suite_name, point))
    pdf = PdfPages(pdf_file_name)
    file_name_template = "{}_f{:03d}_i01_d10_{}.txt"
    for func in range(3, 42, 3):
        files = [os.path.join(path, file_name_template.format(suite_name, func, "reformatted"))]
        plot_line_walk_results(files, pdf=pdf, title="GAN Mario f{} ({} point)".format(func, point),
                               n_rows=3, n_columns=4)
    files = []
    for func in range(3, 42, 3):
        files.append(os.path.join(path, file_name_template.format(suite_name, func, "time")))
    plot_line_walk_times(files, pdf=pdf, title="GAN Mario evaluation times")
    pdf.close()


def rw_top_trumps_line_walk(exdata_path, point='random'):
    suite_name = "rw-top-trumps"
    path = os.path.join(exdata_path, "{}-line-walk-{}".format(suite_name, point))
    reformat_line_walk_results(path)
    pdf_file_name = os.path.join(path, "{}-{}.pdf".format(suite_name, point))
    pdf = PdfPages(pdf_file_name)
    file_name_template = "{}_f{:03d}_i{:02d}_d128_{}.txt"
    for func in range(1, 7):
        files = []
        for instance in range(1, 6):
            files.append(os.path.join(path, file_name_template.format(suite_name, func,
                                                                      instance, "reformatted")))
        plot_line_walk_results(files, pdf=pdf, title="Top trumps f{} ({} point)".format(func, point))
    files = []
    for func in range(1, 7):
        for instance in range(1, 6):
            files.append(os.path.join(path, file_name_template.format(suite_name, func,
                                                                      instance, "time")))
    plot_line_walk_times(files, pdf=pdf, title="Top trumps evaluation times")
    pdf.close()


if __name__ == '__main__':
    exdata_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "code-experiments",
                                               "build", "c", "exdata"))
    rw_gan_mario_line_walk(exdata_path)
    rw_top_trumps_line_walk(exdata_path)
