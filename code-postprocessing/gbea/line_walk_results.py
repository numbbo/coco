from __future__ import absolute_import, division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import ceil, isclose


def reformat_line_walk_results(path):
    """Performs reformatting of the line walk results output from the rw COCO logger in all files
    in path to ease plotting.
    """
    for in_file_name in os.listdir(path):
        out_file_name = in_file_name.replace('_rw', '_reformatted')
        with open(os.path.join(path, in_file_name), 'r') as f_in, \
                open(os.path.join(path, out_file_name), 'w') as f_out:
            dim = None
            previous_line = None
            previous_step = None
            previous_x = 0
            current_x = 0
            current_step = None
            for line in f_in:
                if line[0] == '%' or line.strip() == '':
                    continue
                else:
                    numbers = line.split()
                    if dim is None:
                        # The line containing the origin point
                        dim = len(numbers) - 2
                        f_out.write("# origin\n")
                        for n in numbers[2:]:
                            f_out.write("{}\t{}\t{}\t\n".format(numbers[0], numbers[1], n))
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


def load_line_walk_results(file_name):
    """Returns the line walk results stored in file_name as a dictionary. """
    array = None
    current_set = None
    result_list = {}
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
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][j]
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
        # plt.show()
        if pdf is not None:
            pdf.savefig(fig)
        plt.close()


if __name__ == '__main__':
    exdata_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "code-experiments", "build", "c", "exdata"))

    # Top trumps line walk with the middle point
    if False:
        path = os.path.join(exdata_path, "rw-top-trumps-line-walk-middle")
        reformat_line_walk_results(path)
        pdf_file_name = os.path.join(path, "rw-top-trumps-middle.pdf")
        pdf = PdfPages(pdf_file_name)
        for func in range(1, 6):
            files = [os.path.join(path, "rw-top-trumps_f00{}_i01_d128_reformatted.txt".format(func)),
                     os.path.join(path, "rw-top-trumps_f00{}_i02_d128_reformatted.txt".format(func)),
                     os.path.join(path, "rw-top-trumps_f00{}_i03_d128_reformatted.txt".format(func))]
            plot_line_walk_results(files, pdf=pdf, title="Top trumps f{} (middle point)".format(func))
        pdf.close()

    # Top trumps line walk with the random point
    if False:
        path = os.path.join(exdata_path, "rw-top-trumps-line-walk-random")
        reformat_line_walk_results(path)
        pdf_file_name = os.path.join(path, "rw-top-trumps-random.pdf")
        pdf = PdfPages(pdf_file_name)
        for func in range(1, 6):
            files = [os.path.join(path, "rw-top-trumps_f00{}_i01_d128_reformatted.txt".format(func)),
                     os.path.join(path, "rw-top-trumps_f00{}_i02_d128_reformatted.txt".format(func)),
                     os.path.join(path, "rw-top-trumps_f00{}_i03_d128_reformatted.txt".format(func))]
            plot_line_walk_results(files, pdf=pdf, title="Top trumps f{} (random point)".format(func))
        pdf.close()

    # GAN Mario line walk with the middle point
    if False:
        path = os.path.join(exdata_path, "rw-gan-mario-line-walk-middle")
        reformat_line_walk_results(path)
        pdf_file_name = os.path.join(path, "rw-gan-mario-middle.pdf")
        pdf = PdfPages(pdf_file_name)
        for func in range(3, 42, 3):
            files = [os.path.join(path, "rw-gan-mario_f{0:03d}_i01_d10_reformatted.txt".format(func))]
            plot_line_walk_results(files, pdf=pdf,
                                   title="GAN Mario f{} (middle point)".format(func),
                                   n_rows=3, n_columns=4)
        pdf.close()

    # GAN Mario line walk with the random point
    if True:
        path = os.path.join(exdata_path, "rw-gan-mario-line-walk-random")
        reformat_line_walk_results(path)
        pdf_file_name = os.path.join(path, "rw-gan-mario-random.pdf")
        pdf = PdfPages(pdf_file_name)
        for func in range(3, 22, 3):
            files = [os.path.join(path, "rw-gan-mario_f{0:03d}_i01_d10_reformatted.txt".format(func))]
            plot_line_walk_results(files, pdf=pdf,
                                   title="GAN Mario f{} (random point)".format(func),
                                   n_rows=3, n_columns=4)
        pdf.close()
